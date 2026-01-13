from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from blocksworld_reward_model import BlocksWorldModel
import re
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, PPOConfig
from peft import LoraConfig, get_peft_model
import os
import torch
import time
import fire
import json
from utils import generate_icl
import torch.distributed as dist
import sys

data_files = ['data/blocksworld/train_set-2-more_with_trace.json'] # 'data/blocksworld/train_set-4.json', 'data/blocksworld/train_set-6.json'
THINK = "<think>"
THINK_CLOSE = "</think>"
def prepare_dataset(data_files=data_files):
    dataset = load_dataset('json', data_files=data_files)
    print(dataset)
    dataset = dataset['train'].shuffle(seed=1234).select(range(3000))
    return dataset

def generate_r1_prompt(tokenizer, init, goal, plan = "", example_index=0, icl_examples_set=None):
    if icl_examples_set is None:
        icl_example = ""
    else:
        icl_example = generate_icl(icl_examples_set, provide_think_icl=True, num_icl=1, idx=example_index)
    r1_prefix = [{
                "role": "system",
                "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
        },
        { 
                "role": "user",
                "content": f"I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nHere is the format of the actions: \n\npick up the [block_name] block # for example: pick up the blue block\nunstack the [block_name] block from on top of the [another_block_name] block # for example: unstack the orange block from on top of the black block\nput down the [block_name] block # for example put down the red block\nstack the [block_name] block on top of the [another_block_name] block # for example: stack the yellow block on top of the red block \n\n{icl_example}\n\n[Problem]\nHere is the initial state of the blocks: {init}\n\nHere is the goal state of the blocks: {goal}.\nShow your work in the {THINK} {THINK_CLOSE} tags. Return the final sequence of actions as the plan in the <plan> </plan> tags.\n" # , for example: <plan>\npick up the blue block\nstack the blue block on top of the yellow block\nunstack the orange block from on top of the black block\nstack the orange block on top of the red block</plan>
        },
        {
                "role": "assistant",
                "content": f"Let me solve this step by step.\n{THINK}"
        }
    ]
    # print(len(tokenizer.encode(tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True))))
    return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "plan": plan, "init": init, "goal": goal}

def validate_response_format(response: str):
    # Remove leading/trailing whitespace
    response = response.strip()
    
    # Rule 1: Must start with <think> and end with </plan>
    if not response.startswith(THINK) or not response.endswith("</plan>"):
        print(f'Response does not start with {THINK} or end with </plan>')
        return False

    # Rule 2: Must contain exactly one of each tag.
    if response.count(THINK) != 1 or response.count(THINK_CLOSE) != 1:
        print('Response does not contain exactly one of each think tag')
        return False
    if response.count("<plan>") != 1 or response.count("</plan>") != 1:
        print('Response does not contain exactly one of each plan tag')
        return False
    
    # Find indices for each tag.
    think_open = response.find(THINK)
    think_close = response.find(THINK_CLOSE)
    plan_open = response.find("<plan>")
    plan_close = response.find("</plan>")

    # Rule 4: The order should be: <think> ... </think> then <plan> ... </plan>
    if think_open != 0:  # Should start with <think>
        print(f'Response does not start with {THINK}')
        return False
    if think_close == -1 or plan_open == -1 or plan_close == -1:
        print(f'Response does not contain <plan> and </plan>, or {THINK_CLOSE}')
        return False
    if think_close > plan_open:
        print('Response has closing think tag after opening plan tag')
        return False  # The closing think tag must come before the opening plan tag
    
    # Rule 3: Check non-empty content between tags.
    think_content = response[len(THINK):think_close].strip()
    plan_content = response[plan_open + len("<plan>"):plan_close].strip()

    if not think_content or not plan_content:
        return False

    return True

def reward_fn_wrapper(completions, plan, init, goal, **kwargs):
    rewards = []
    for completion, plan_i, init_i, goal_i in zip(completions, plan, init, goal):
        reward_format = 0.0
        try:
            print('#########################')
            completion = THINK + completion
            print(completion)
            if not validate_response_format(completion):
                print('Response Format Error')
                rewards.append(0.0) # Penalty to avoid format errors. Rewards will be only for the plan.
                continue
            else:
                reward_format = 1.0
            # Second reward for extracting the plan and physical feasibility
            matches = re.findall(r"<plan>(.*?)</plan>", completion, flags=re.DOTALL | re.IGNORECASE)
            if matches is None or len(matches) != 1:
                print("No plan found")
                rewards.append(0.0)
                continue
            
            # Process plan
            non_empty = [match.strip() for match in matches if match.strip()] # Ideally, we should have only one match
            extracted_plan = non_empty[0]
            
            
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
                            
"""
Pick up D from B
Unstack D from B
Stack D on A
Pick up C
Stack C on B
Pick up B
Stack B on C


We start from the initial state: A on C, D on B, C on table, B on table. We want D on A and C on B.
"""            
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
    target_bytes = gb * 1024**3  
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


def main(use_icl_examples=False):
    try:
        hf_token = os.environ.get("hf_token")
        # login(token=hf_token, add_to_git_credential=False) # ADD YOUR TOKEN HERE

        # Model config
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", torch_dtype="bfloat16")
        model_config = ModelConfig(
            model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
            # use_peft=True,
            # load_in_4bit=True,
            lora_task_type="CAUSAL_LM",
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            lora_target_modules=["q_proj", "v_proj"],
        )
        # lora_config = LoraConfig(
        #     r=32,
        #     lora_alpha=64,
        #     target_modules=["q_proj", "v_proj"],
        #     lora_dropout=0.1,
        #     task_type="CAUSAL_LM",
        # )
        # model = get_peft_model(model, lora_config)
        # model.print_trainable_parameters()
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        if use_icl_examples:
            with open('data/blocksworld/train_set-2-more_with_trace.json') as f:
                icl_examples = json.load(f)
        else:
            icl_examples = None
        # Format the dataset
        dataset = prepare_dataset()
        dataset = dataset.map(lambda example, idx: generate_r1_prompt(tokenizer, example["init"], example["goal"], example["plan"], idx, icl_examples), with_indices=True)
        train_test_split = dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]

        # Hyperparameters
        print('herererere')
        training_args = GRPOConfig(
            output_dir="qwen-bw-r1-aha-moment/deep_seek-r1-2step-sing",
            learning_rate=1e-6,
            lr_scheduler_type="cosine",
            logging_steps=10,
            max_steps=400,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            # gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True,
            # GRPO specific parameters
            max_prompt_length=1600,
            max_completion_length=512, # max length of the generated output for our solution
            num_generations=8,
            beta=0.001,
            # temperature=0.5,
            # Vllm
            use_vllm=True,
            vllm_gpu_memory_utilization=0.5,
            # Reporting
            report_to=["tensorboard"],
            push_to_hub=True,
            save_strategy="steps",
            save_steps=10,   
        )
        trainer = GRPOTrainer(
            model=model_config.model_name_or_path,
            reward_funcs=[reward_fn_wrapper],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=get_peft_config(model_config),
        )

        trainer.train()
        trainer.save_model(training_args.output_dir)
        trainer.push_to_hub(dataset_name='bw-test')
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught. Shutting down distributed processes gracefully...")
        # Clean up the distributed environment if it was initialized
        if torch.distributed.is_initialized():
            print('Shutting Distributed Process')
            dist.destroy_process_group()
        sys.exit(0)

# occupy_gpu_memory(gb=50, device="cuda:0")

if __name__ == "__main__":
    fire.Fire(main)
