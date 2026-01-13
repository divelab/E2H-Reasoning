"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import re
import os
from datasets import Dataset, load_dataset, DatasetDict
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
import argparse
import operator
import random
from pathlib import Path
from tqdm import trange


def gen_dataset(
        num_samples: int,
        num_operands: int = 6,
        max_target: int = 1000,
        min_number: int = 1,
        max_number: int = 100,
        operations: List[str] = ['+', '-', '*', '/'],
        seed_value: int = 42,
) -> List[Tuple]:
    """Generate dataset for countdown task with valid targets.

    This function generates a dataset where each target is guaranteed
    to be achievable using the provided numbers and operations.
    All intermediate results are ensured to be integers.

    Args:
        num_samples: Number of samples to generate
        num_operands: Number of numbers provided in each sample
        max_target: Maximum value for target number
        min_number: Minimum value for provided numbers
        max_number: Maximum value for provided numbers
        operations: List of allowed operations
        seed_value: Random seed for reproducibility

    Returns:
        List of tuples containing (target, numbers, valid_solution)
    """
    # Set random seed for reproducibility
    random.seed(seed_value)
    samples = []

    for _ in tqdm(range(num_samples)):
        # Generate random numbers
        numbers = [random.randint(min_number, max_number) for _ in range(num_operands)]

        # Generate a valid target and solution
        target = None
        valid_solution = None
        max_attempts = 40

        for attempt in range(max_attempts):
            # Shuffle numbers and create a copy
            nums_copy = numbers.copy()
            random.shuffle(nums_copy)

            # Start with first number
            if not nums_copy:
                continue

            current_value = nums_copy[0]
            expression = str(nums_copy[0])
            reasoning_steps = [f'{current_value}']
            remaining_nums = nums_copy[1:]

            # Process each remaining number
            for num in remaining_nums:
                # Choose a random operation
                shuffled_ops = operations.copy()
                random.shuffle(shuffled_ops)

                # Try operations until we find one that gives an integer result
                for op in shuffled_ops:
                    if op == '+':
                        new_value = current_value + num
                        new_expr = f"({expression} + {num})"
                        reasoning_steps.append(f"{current_value} + {num} = {new_value}")
                        break
                    elif op == '-':
                        new_value = current_value - num
                        new_expr = f"({expression} - {num})"
                        reasoning_steps.append(f"{current_value} - {num} = {new_value}")
                        break
                    elif op == '*':
                        new_value = current_value * num
                        new_expr = f"({expression} * {num})"
                        reasoning_steps.append(f"{current_value} * {num} = {new_value}")
                        break
                    elif op == '/':
                        # Check for division by zero and integer division
                        if num != 0 and current_value % num == 0:
                            new_value = current_value // num
                            new_expr = f"({expression} / {num})"
                            reasoning_steps.append(f"{current_value} / {num} = {new_value}")
                            break

                current_value = new_value
                expression = new_expr

            # Check if we have a valid solution within range
            if 0 <= current_value <= max_target:
                target = current_value
                valid_solution = expression
                reasoning_steps.append(f"{current_value}")
                break

        # If we couldn't find a valid expression after max attempts, create a simple one
        if target is not None:
            sample = {
                "target": target,
                "nums": numbers,
                "expression": valid_solution,
                "reasoning_steps": reasoning_steps,
            }
            samples.append(sample)

    return samples



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--local_dir', default='~/data/countdown')
    # parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=500000)
    parser.add_argument('--num_operands', type=int, default=6)
    parser.add_argument('--max_target', type=int, default=1000)
    parser.add_argument('--min_number', type=int, default=1)
    parser.add_argument('--max_number', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=327680)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'countdown'

    dataset_name = f'{data_source}_n{args.num_operands}t{args.max_target}_{args.min_number}-{args.max_number}'
    # local_dir = Path(args.local_dir) / dataset_name
    # local_dir.mkdir(exist_ok=True)

    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')
    all_samples = gen_dataset(num_samples=args.num_samples,
                              num_operands=args.num_operands,
                             max_target=args.max_target,
                             min_number=args.min_number,
                             max_number=args.max_number)
    raw_dataset = Dataset.from_list(all_samples)

    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            solution = {
                "target": example['target'],
                "numbers": example['nums'],
                "expression": example['expression']
            }
            data = {
                "data_source": data_source,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    # raw_dataset = raw_dataset.map(function=make_map_fn('raw'), with_indices=True)

    # hdfs_dir = args.hdfs_dir

    combined_dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    # raw_dataset.save_to_disk(local_dir / 'countdown_dataset')
    combined_dataset.push_to_hub(f"{dataset_name}")
    # train_dataset.to_parquet(local_dir/ 'train.parquet')
    # test_dataset.to_parquet(local_dir / 'test.parquet')

    # if hdfs_dir is not None:
    #     makedirs(hdfs_dir)
    #     copy(src=local_dir, dst=hdfs_dir)
