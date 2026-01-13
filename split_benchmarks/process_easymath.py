import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets
import numpy as np
import json
import os
from tqdm import tqdm 
import torch
import re 
from reasoners.lm.hf_model import HFModel
import random 

def extract_answer(text):
    if not text:
        return None
    match = re.search(r"####([\d,\.\-]+)####", text)
    if match:
        ans_str = match.group(1).replace(",", "")
        try:
            return float(ans_str)
        except ValueError:
            pass
    matches = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None

def calculate_difficulty(solution, model, question, num_samples=20):
    prompt = f"Solve the following math problem step-by-step. Only provide the final numerical answer in the format ####number#### \n\nQuestion: {question}\nAnswer:"
    print(prompt)
    prompts = [prompt] * num_samples
    outputs = model.generate(prompts,
                                  hide_input=True,
                                  do_sample=True,
                                  temperature=0.8,
                                  max_length=512).text

    extracted_answers = [extract_answer(output) for output in outputs]

    # Compute the number of correct answers
    num_correct = sum(ans == solution for ans in extracted_answers)
    print(extracted_answers)
    print(num_correct)

    return num_correct

def save_split_to_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def main():
    base_model = HFModel("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", quantized='bf16', max_batch_size=32)

    print("Loading easy math dataset (main config)...")

    data = load_jsonl("/mnt/data/shared/blakeo/LM-Reasoning/Reasoning/split_benchmarks/TEMH_splits/easymath_aqua_gsm8k_splits/trivial_easymath.jsonl")
    print("Calculating difficulty scores...")
    difficulties = []
    for example in tqdm(data, desc="Calculating difficulties", unit="example"):
        difficulty = calculate_difficulty(float(example['answer']), base_model, example['question'])
        difficulties.append(difficulty)

    q1 = np.quantile(difficulties, 0.25)  # 25th percentile
    q2 = np.quantile(difficulties, 0.50)  # 50th percentile (median)
    q3 = np.quantile(difficulties, 0.75)  # 75th percentile

    trivial_split = []
    easy_split = []
    medium_split = []
    hard_split = []

    print("Splitting dataset into quartiles...")
    for i, example in enumerate(data):
        difficulty = difficulties[i]
        example_with_difficulty = example.copy()
        example_with_difficulty['difficulty_score'] = difficulty

        if difficulty <= q1:
            hard_split.append(example_with_difficulty)
        elif difficulty <= q2:
            medium_split.append(example_with_difficulty)
        elif difficulty <= q3:
            easy_split.append(example_with_difficulty)
        else:
            trivial_split.append(example_with_difficulty)

    print(f"Split sizes: Trivial={len(trivial_split)}, Easy={len(easy_split)}, Medium={len(medium_split)}, Hard={len(hard_split)}")

    output_dir = "sc_splits/easymath_sc_splits"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving splits to '{output_dir}' directory...")
    save_split_to_jsonl(trivial_split, os.path.join(output_dir, "trivial.jsonl"))
    save_split_to_jsonl(easy_split, os.path.join(output_dir, "easy.jsonl"))
    save_split_to_jsonl(medium_split, os.path.join(output_dir, "medium.jsonl"))
    save_split_to_jsonl(hard_split, os.path.join(output_dir, "hard.jsonl"))

    print("Splitting complete.")

if __name__ == "__main__":
    main()
