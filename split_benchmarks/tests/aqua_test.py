import json
import os
import re
import torch
from tqdm import tqdm
import numpy as np
from reasoners.lm.hf_model import HFModel
import argparse


import re

def extract_answer(text):
    if not text:
        return None
    match = re.search(r"####\s*([A-Ea-e])\s*####", text)
    if match:
        return match.group(1).upper()

    # Fallback in case there's no ####A#### format but still a letter answer somewhere
    matches = re.findall(r"\b([A-Ea-e])\b", text)
    if matches:
        return matches[-1].upper()

    return None


def calculate_accuracy(data, model, num_samples=20):
    correct_total = 0
    total = len(data)
    
    i = 1
    for example in tqdm(data, desc="Evaluating", unit="example"):
        question = example['question']
        solution = example['correct']
        choices = example['options']
        prompt = f"Solve the following multiple choice math problem step-by-step. Only provide the final letter answer in the format ####answer#### \n\nQuestion: {question} {choices}\nAnswer:"
        prompts = [prompt]
        
        outputs = model.generate(
            prompts,
            hide_input=True,
            do_sample=True,
            temperature=0.8,
            max_length=1024
        ).text

        extracted_answer = extract_answer(outputs[0])
        if (extracted_answer == solution):
            correct_total += 1

        print(correct_total / i)
        i += 1

    accuracy = correct_total / total if total > 0 else 0
    return accuracy, correct_total, total

def load_data(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]

def main():
    parser = argparse.ArgumentParser(description="Evaluate accuracy on a specified math difficulty split.")
    parser.add_argument('--split', type=str, required=True, choices=['trivial', 'easy', 'medium', 'hard'],
                        help="Which difficulty split to evaluate.")

    args = parser.parse_args()
    base_model = HFModel("Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B", quantized='bf16', max_batch_size=32)

    split_paths = {
        "trivial": "TEMH_splits/aqua_splits/aqua_trivial.jsonl",
        "easy": "TEMH_splits/aqua_splits/aqua_easy.jsonl",
        "medium": "TEMH_splits/aqua_splits/aqua_medium.jsonl",
        "hard": "TEMH_splits/aqua_splits/aqua_hard.jsonl"
    }

    print(f"Evaluating split: {args.split}")

    data = load_data(split_paths[args.split])
    accuracy, correct_total, total = calculate_accuracy(data, base_model)

    result = {args.split: accuracy, "num_correct": correct_total, "total": total}
    print(f"Accuracy on {args.split} split: {accuracy:.2%}")

    # Save results to file
    with open(f"{args.split}_results", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to '{args.split}_results'")

if __name__ == "__main__":
    main()
