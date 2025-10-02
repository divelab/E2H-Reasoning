import os
import sys
import hydra
import torch
from pprint import pprint
from omegaconf import OmegaConf
from transformers import set_seed

from src.datasets import get_dataset
from src.rewards import get_reward_fn


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank()==0):
        os.makedirs(cfg.output_dir, exist_ok=True)
        sys.stdout = open(os.path.join(cfg.output_dir, "log.txt"), "a")
    else:
        sys.stdout = open(os.devnull, "w")

    print("\n\nConfig:")
    pprint(OmegaConf.to_container(cfg, resolve=True), indent=4, width=2)
        
    set_seed(cfg.algorithm.args.seed)

    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "test":
        test(cfg)


def train(cfg):

    from trl import ModelConfig, get_peft_config
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # from src.trainer import CurriculumGRPOConfig, CurriculumGRPOTrainer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.args.model_name_or_path,
        trust_remote_code=cfg.model.args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.args.model_name_or_path,
        dtype=cfg.model.args.dtype,
        trust_remote_code=cfg.model.args.trust_remote_code,
        attn_implementation=cfg.model.args.attn_implementation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("\n\nModel:")
    print(model)

    dataset = get_dataset(
        cfg.task.args,
        tokenizer,
        cfg.algorithm.args.seed
    )
    print("\n\nDataset:")
    print(dataset)
    print("Train Difficulty Levels: ", sorted(list(set(dataset["train"]["level"]))))
    print("Test Difficulty Levels: ", sorted(list(set(dataset["test"]["level"]))), "\n\n")

    reward_fn = get_reward_fn(
        cfg.task.args
    )

    if cfg.alogorithm.name == "GRPO":
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            args=GRPOConfig(**cfg.algorithm.args),
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=tokenizer,
            peft_config=get_peft_config(ModelConfig(**cfg.model.args))
        )
    else:
        raise f"{cfg.alogorithm.name} Trainer not Implemented"

    trainer.train()
    trainer.save_model(cfg.algorithm.output_dir)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def test(cfg):
    
    import numpy as np
    from vllm import LLM, SamplingParams
    
    model = LLM(
        model=cfg.algorithm.args.output_dir,
        trust_remote_code=cfg.model.args.trust_remote_code,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype=cfg.model.args.dtype,
        max_model_len=cfg.task.args.max_prompt_length+cfg.task.args.max_completion_length,
        seed=cfg.algorithm.args.seed,
        task='generate'
    )

    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        max_tokens=cfg.task.args.max_completion_length,
        min_tokens=1,
        seed=cfg.algorithm.args.seed,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    dataset = get_dataset(
        cfg.task.args,
        tokenizer,
        cfg.algorithm.args.seed
    )['test']

    outputs = model.generate(dataset['prompt'], sampling_params)
    outputs = [
        completion_output.text
        for request_output in outputs
        for completion_output in request_output.outputs
    ]
    dataset = dataset.add_column('output', outputs)

    reward_fn = get_reward_fn(
        cfg.task.args
    )
    rewards = np.array(
        reward_fn(
            completions=dataset['output'],
            answer=dataset['answer']
        )
    )
    dataset = dataset.add_column('reward', rewards.tolist())
    dataset.to_json(os.path.join(cfg.algorithm.args.output_dir, 'test_outputs.jsonl'))

    results = dict()
    results['overall'] = {
        'accuracy': (rewards > cfg.task.args.format_reward).mean().item(),
        'support': len(dataset)
    }
    for level in sorted(list(set(dataset["level"]))):
        level_outputs = dataset.filter(lambda example: example['level']==level)
        level_rewards = np.array(level_outputs['reward'])
        results[f"level_{level}"] = {
            'accuracy': (level_rewards > cfg.task.args.format_reward).mean().item(),
            'support': len(level_rewards)
        }
    with open(os.path.join(cfg.algorithm.args.output_dir, 'test_results.json'), "w") as f:
        json.dump(results, f, indent=4)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()