import os
import re
import time
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import json
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig

from blocksworld_reward_model import BlocksWorldModel
from utils import generate_icl


class BlocksWorldTrainer:
    """Class for training blocksworld models using Hydra for configuration"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # Setting up paths
        self.output_dir = Path(HydraConfig.get().run.dir)  # Hydra changes working directory

        # Save the config for reproducibility
        with open(self.output_dir / "config_dump.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

        root_path = Path(os.environ['ROOT_PATH'])
        os.chdir(root_path)
        print(f"Working directory: {root_path}")
        print(f"Output directory: {self.output_dir}")

        # Setup HuggingFace authentication
        hf_token = self.cfg.training.hf_token
        if hf_token:
            login(token=hf_token, add_to_git_credential=True)
        else:
            print("Warning: No HuggingFace token provided. Some operations may fail.")

    def train(self):
        """Train a model using GRPO with configurations from Hydra"""
        # Extract config values
        model_name = self.cfg.model.name
        use_icl_examples = self.cfg.experiment.use_icl_examples
        output_model_name = self.cfg.output.model_name

        # Prepare ICL examples if needed
        icl_examples = None
        if use_icl_examples:
            with open(self.cfg.experiment.icl_examples_file) as f:
                icl_examples = json.load(f)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.cfg.model.trust_remote_code
        )

        # Prepare dataset
        dataset = self._prepare_dataset()
        dataset = dataset.map(
            lambda example, idx: self._generate_r1_prompt(
                tokenizer,
                example["init"],
                example["goal"],
                example["plan"],
                idx,
                icl_examples
            ),
            with_indices=True
        )

        # Split dataset
        train_test_split = dataset.train_test_split(test_size=self.cfg.experiment.test_size)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]

        # Setup Model config
        model_config = self._get_model_config()

        # Setup training arguments
        training_args = self._get_training_config()

        # Create trainer
        trainer = GRPOTrainer(
            model=model_config.model_name_or_path,
            reward_funcs=[self._reward_fn_wrapper],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=get_peft_config(model_config),
        )

        # Train model
        trainer.train()
        trainer.save_model(training_args.output_dir)

        if self.cfg.training.push_to_hub:
            trainer.push_to_hub(dataset_name='blocksworld-dataset')

    def _prepare_dataset(self):
        """Prepare dataset for training"""
        dataset = load_dataset('json', data_files=self.cfg.experiment.data_files)
        dataset = dataset['train'].shuffle(seed=self.cfg.experiment.dataset_seed)

        # Limit dataset size if specified
        if self.cfg.experiment.dataset_size > 0:
            dataset = dataset.select(range(self.cfg.experiment.dataset_size))

        print(f"Dataset prepared with {len(dataset)} samples")
        return dataset

    def _get_model_config(self):
        """Create model configuration"""
        lora_config = self.cfg.lora

        model_config = ModelConfig(
            model_name_or_path=self.cfg.model.name,
            torch_dtype=self.cfg.model.torch_dtype,
            attn_implementation=self.cfg.model.attn_implementation,
            lora_task_type=lora_config.task_type,
            lora_r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            lora_target_modules=list(lora_config.target_modules),
        )

        return model_config

    def _get_training_config(self):
        """Create training configuration"""
        training_cfg = self.cfg.training
        output_dir = self.output_dir / self.cfg.output.model_name

        training_args = GRPOConfig(
            output_dir=str(output_dir),
            learning_rate=training_cfg.learning_rate,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            logging_steps=training_cfg.logging_steps,
            max_steps=training_cfg.max_steps,
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            gradient_checkpointing=training_cfg.gradient_checkpointing,
            bf16=training_cfg.bf16,
            # GRPO specific parameters
            max_prompt_length=training_cfg.max_prompt_length,
            max_completion_length=training_cfg.max_completion_length,
            num_generations=training_cfg.num_generations,
            beta=training_cfg.beta,
            # Vllm
            use_vllm=training_cfg.use_vllm,
            vllm_gpu_memory_utilization=training_cfg.vllm_gpu_memory_utilization,
            # Reporting
            report_to=list(training_cfg.report_to),
            push_to_hub=training_cfg.push_to_hub,
            save_strategy=training_cfg.save_strategy,
            save_steps=training_cfg.save_steps,
        )

        return training_args

    def _generate_r1_prompt(self, tokenizer, init, goal, plan="", example_index=0, icl_examples_set=None):
        """Generate prompt for the model"""
        if icl_examples_set is None:
            icl_example = ""
        else:
            icl_example = generate_icl(icl_examples_set, provide_think_icl=True, num_icl=1, idx=example_index)

        r1_prefix = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
            },
            {
                "role": "user",
                "content": f"I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nHere is the format of the actions: \n\npick up the [block_name] block # for example: pick up the blue block\nunstack the [block_name] block from on top of the [another_block_name] block # for example: unstack the orange block from on top of the black block\nput down the [block_name] block # for example put down the red block\nstack the [block_name] block on top of the [another_block_name] block # for example: stack the yellow block on top of the red block \n\n{icl_example}\n\n[Problem]\nHere is the initial state of the blocks: {init}\n\nHere is the goal state of the blocks: {goal}.\nShow your work in the <think> </think> tags. Return the final sequence of actions as the plan in the <plan> </plan> tags.\n"
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>"
            }
        ]

        return {
            "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
            "plan": plan,
            "init": init,
            "goal": goal
        }

    def _validate_response_format(self, response: str):
        """Validate the response format"""
        # Remove leading/trailing whitespace
        response = response.strip()

        # Rule 1: Must start with <think> and end with </plan>
        if not response.startswith("<think>") or not response.endswith("</plan>"):
            print('Response does not start with <think> or end with </plan>')
            return False

        # Rule 2: Must contain exactly one of each tag.
        if response.count("<think>") != 1 or response.count("</think>") != 1:
            print('Response does not contain exactly one of each think tag')
            return False
        if response.count("<plan>") != 1 or response.count("</plan>") != 1:
            print('Response does not contain exactly one of each plan tag')
            return False

        # Find indices for each tag.
        think_open = response.find("<think>")
        think_close = response.find("</think>")
        plan_open = response.find("<plan>")
        plan_close = response.find("</plan>")

        # Rule 4: The order should be: <think> ... </think> then <plan> ... </plan>
        if think_open != 0:  # Should start with <think>
            print('Response does not start with <think>')
            return False
        if think_close == -1 or plan_open == -1 or plan_close == -1:
            print('Response does not contain <plan> and </plan>, or </think>')
            return False
        if think_close > plan_open:
            print('Response has closing think tag after opening plan tag')
            return False  # The closing think tag must come before the opening plan tag

        # Rule 3: Check non-empty content between tags.
        think_content = response[len("<think>"):think_close].strip()
        plan_content = response[plan_open + len("<plan>"):plan_close].strip()

        if not think_content or not plan_content:
            return False

        return True

    def _reward_fn_wrapper(self, completions, plan, init, goal, **kwargs):
        """Wrapper for reward function"""
        rewards = []
        for completion, plan_i, init_i, goal_i in zip(completions, plan, init, goal):
            reward_format = 0.0
            try:
                print('#########################')
                completion = "<think>" + completion
                print(completion)

                if not self._validate_response_format(completion):
                    print('Response Format Error')
                    rewards.append(0.0)  # Penalty to avoid format errors
                    continue
                else:
                    reward_format = 1.0

                # Extract the plan
                matches = re.findall(r"<plan>(.*?)</plan>", completion, flags=re.DOTALL | re.IGNORECASE)
                if matches is None or len(matches) != 1:
                    print("No plan found")
                    rewards.append(0.0)
                    continue

                # Process plan
                non_empty = [match.strip() for match in matches if
                             match.strip()]  # Ideally, we should have only one match
                extracted_plan = non_empty[0]

                # Calculate reward
                instance_example = BlocksWorldModel(init_i, goal_i, extracted_plan)
                reward = instance_example.simulate_plan_with_reward(true_plan=plan_i) + reward_format
                rewards.append(reward)
                print('-----')
                print(reward)
                print(init_i)
                print(goal_i)
                print('-----')
                print('#########################')
            except Exception as e:
                print(e)
                rewards.append(0.0)

        return rewards


def occupy_gpu_memory(gb=75, device="cuda:0"):
    """
    Allocates a tensor on the specified GPU that occupies approximately `gb` GB of memory.
    The tensor remains allocated indefinitely (until the process is terminated).

    Parameters:
      - gb (int): Target memory in gigabytes (default: 75).
      - device (str): The CUDA device (e.g. "cuda:0") to allocate on.

    Note: If you set CUDA_VISIBLE_DEVICES=7,2 then "cuda:0" is physical GPU 7.
    """
    # Calculate the target memory in bytes.
    target_bytes = gb * 1024 ** 3
    # For float32, each element takes 4 bytes.
    num_elements = target_bytes // 4
    torch.cuda.empty_cache()
    print(f"Allocating a tensor with {num_elements} float32 elements (~{gb}GB) on {device}.")

    try:
        # Allocate the tensor on the specified device.
        tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
        tensor.fill_(0)
        print(f"Successfully allocated ~{gb}GB on {device}. Holding memory indefinitely...")
    except RuntimeError as e:
        print("Failed to allocate memory. Your GPU may not have enough free memory.")
        raise e

    # Hold the memory indefinitely.
    while True:
        print("Holding memory...")
        time.sleep(60)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Main entry point for training with Hydra configuration"""
    print(OmegaConf.to_yaml(cfg))
    trainer = BlocksWorldTrainer(cfg)
    trainer.train()

    # Optional: Occupy GPU memory after training (useful for server environments)
    # occupy_gpu_memory(gb=50, device="cuda:0")


if __name__ == "__main__":
    main()