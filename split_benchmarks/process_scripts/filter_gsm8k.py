import datasets
import json
import os
from tqdm import tqdm

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Save data to a JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    print("Loading GSM8K test dataset...")
    gsm8k_train = datasets.load_dataset("gsm8k", "main", split="train")
    gsm8k_test = datasets.load_dataset("gsm8k", "main", split="test")
    print("length of gsm8k: ", len(gsm8k_test))
    
    test_questions = set(example['question'] for example in gsm8k_test)
    train_questions = set(example['question'] for example in gsm8k_train)

    print(f"Loaded {len(test_questions)} test questions")
    
    input_dir = "split_benchmarks/sc_splits/gsm8k_sc_splits"
    output_dir = "split_benchmarks/sc_splits/gsm8k_sc_splits"
    os.makedirs(output_dir, exist_ok=True)

    total_filtered = 0
    total_not_in_train = 0
    for difficulty in ["trivial", "easy", "medium", "hard"]:
        input_file = os.path.join(input_dir, f"gsm8k_{difficulty}.jsonl")
        output_file = os.path.join(output_dir, f"gsm8k_{difficulty}.jsonl")
        
        print(f"Processing {difficulty} split...")
        
        try:
            data = load_jsonl(input_file)
            print(f"  Loaded {len(data)} examples")
        except FileNotFoundError:
            print(f"  Warning: {input_file} not found, skipping...")
            continue
        
        total_filtered += len(data)
        train_only_data = [item for item in data if item['question'] not in test_questions]
        not_in_train = sum(1 for item in data if item['question'] not in train_questions)

        total_not_in_train += not_in_train
        total_filtered -= len(train_only_data)

        print(f"  Filtered to {len(train_only_data)} training examples")
        
        save_jsonl(train_only_data, output_file)
        print(f"  Saved to {output_file}")

    print("total not in train: ", total_not_in_train)
    print("total filtered:", total_filtered)
    print("Filtering complete.")

if __name__ == "__main__":
    main()