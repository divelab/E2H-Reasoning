from datasets import load_dataset
import json
import os

# Load GSM8K dataset splits
dataset = load_dataset("gsm8k", "main")
# Collect set of test questions
test_questions = set(entry["question"].strip() for entry in dataset["test"])

# Define input files and their base names
input_files = ["split_benchmarks/GSM8K_FULL/TEMH/trivial.jsonl", "split_benchmarks/GSM8K_FULL/TEMH/easy.jsonl", "split_benchmarks/GSM8K_FULL/TEMH/medium.jsonl", "split_benchmarks/GSM8K_FULL/TEMH/hard.jsonl"]

for infile in input_files:
    base = os.path.splitext(infile)[0]
    train_out = f"{base}_train.jsonl"
    test_out = f"{base}_test.jsonl"

    with open(infile, 'r', encoding='utf-8') as fin, \
         open(train_out, 'w', encoding='utf-8') as ftrain, \
         open(test_out, 'w', encoding='utf-8') as ftest:
        for line in fin:
            entry = json.loads(line)
            q = entry.get("question", "").strip()
            # Split based on presence in test_questions
            if q in test_questions:
                ftest.write(json.dumps(entry, ensure_ascii=False) + "\n")
            else:
                ftrain.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"{split.capitalize():<7} → train: {train_count:<4}  test: {test_count:<4}")
    print(f"Processed {infile}: created {train_out} and {test_out}")
