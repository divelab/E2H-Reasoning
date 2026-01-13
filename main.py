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
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "test":
        test(cfg)


def train(cfg):

    from trl import GRPOConfig
    from accelerate import Accelerator
    from trl import ModelConfig, get_peft_config
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from src.trainer import CurriculumGRPOTrainer

    if Accelerator().is_main_process:
        os.makedirs(cfg.algorithm.args.output_dir, exist_ok=True)
        sys.stdout = sys.stderr = open(os.path.join(cfg.algorithm.args.output_dir, "main.log"), "a")
    else:
        sys.stdout = open(os.devnull, "w")
        
    set_seed(cfg.algorithm.args.seed)

    print("\n\nConfig:")
    pprint(OmegaConf.to_container(cfg, resolve=True), indent=4, width=2)

    print("\n\nModel:")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.args.model_name_or_path,
        dtype=cfg.model.args.torch_dtype,
        trust_remote_code=cfg.model.args.trust_remote_code,
        attn_implementation=cfg.model.args.attn_implementation
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.args.model_name_or_path,
        trust_remote_code=cfg.model.args.trust_remote_code
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(model)

    print("\n\nDataset:")
    dataset = get_dataset(
        cfg.task.args,
        tokenizer,
        cfg.algorithm.args.seed
    )
    print(dataset)
    print("Train Difficulty Levels: ", sorted(list(set(dataset["train"]["level"]))))
    print("Test Difficulty Levels: ", sorted(list(set(dataset["test"]["level"]))), "\n\n")

    print("\n\nTraining:")
    if cfg.algorithm.name == "GRPO":
        trainer = CurriculumGRPOTrainer(
            model=model,
            reward_funcs=get_reward_fn(cfg.task.args),
            args=GRPOConfig(**OmegaConf.to_container(cfg.algorithm.args, resolve=True)),
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=tokenizer,
            peft_config=get_peft_config(ModelConfig(**OmegaConf.to_container(cfg.model.args, resolve=True))),
            scheduler_params=cfg.algorithm.e2h_args
        )
    else:
        raise f"{cfg.alogorithm.name} Trainer not Implemented"

    trainer.train()
    trainer.save_model()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def test(cfg):
    
    import json
    import numpy as np
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    os.makedirs(cfg.algorithm.args.output_dir, exist_ok=True)
    sys.stdout = sys.stderr = open(os.path.join(cfg.algorithm.args.output_dir, "main.log"), "a")

    set_seed(cfg.algorithm.args.seed)
    
    print("\n\n\n\nTesting:")

    model = LLM(
        model=cfg.model.args.model_name_or_path,
        trust_remote_code=cfg.model.args.trust_remote_code,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype=cfg.model.args.torch_dtype,
        seed=cfg.algorithm.args.seed,
        max_model_len=cfg.task.args.max_prompt_length+cfg.task.args.max_completion_length,
        task='generate',
        enable_lora=True,
        max_lora_rank=64
    )
    tokenizer = model.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

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

    outputs = model.generate(
        dataset['prompt'], 
        sampling_params,
        lora_request=LoRARequest("lora_adapter", 1, cfg.algorithm.args.output_dir) if cfg.model.args.use_peft else None
    )
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
            answer=dataset['answer'],
            question=dataset['question'],
            solution=dataset['solution']
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

    print("\n\nResults:")
    pprint(results, indent=4, width=2)
    with open(os.path.join(cfg.algorithm.args.output_dir, 'test_results.json'), "w") as f:
        json.dump(results, f, indent=4)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()