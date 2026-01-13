import json
import os
import numpy as np

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line.strip()) for line in f]

def save_jsonl(data, filename):
    with open(filename, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_data(dataset_dir, source):
    data = []

    for filename in os.listdir(dataset_dir):
      if filename.endswith(".jsonl"):
          file_path = os.path.join(dataset_dir, filename)
          file_data = load_jsonl(file_path)
          for item in file_data:
            item["source"] = source
          data.extend(file_data)

    return data

def count_sources(split):
    counts = {"gsm8k": 0, "aqua": 0, "other": 0}
    for item in split:
        counts[item["source"]] += 1
    return counts

def normalize_keys(data):
    """Normalize inconsistent keys in the data."""
    normalized_data = []
    
    for item in data:
        # If the item has a "correct" key, rename it to "answer"
        if "correct" in item:
            item["answer"] = item.pop("correct")
        normalized_data.append(item)
    
    return normalized_data

def main():
    data = []
    data.extend(load_data("/mnt/data/shared/blakeo/LM-Reasoning/Reasoning/split_benchmarks/GSM8K_FULL", "gsm8k"))
    print(f"Total combined examples: {len(data)}")
    data = normalize_keys(data)

    # Extract difficulty scores
    difficulty_scores = [item['difficulty_score'] for item in data]

    # Compute new quantiles for three categories
    q1 = np.quantile(difficulty_scores, 1/3)  # 33.33rd percentile
    q2 = np.quantile(difficulty_scores, 2/3)  # 66.67th percentile

    easy, medium, hard = [], [], []

    print("Re-splitting combined data based on updated quantiles...")
    for item in data:
        score = item['difficulty_score']
        if score <= q1:
            hard.append(item)  # Lower scores mean harder problems
        elif score <= q2:
            medium.append(item)
        else:
            easy.append(item)  # Higher scores mean easier problems

    print(f"New split sizes: Easy={len(easy)}, Medium={len(medium)}, Hard={len(hard)}")

    output_dir = "/mnt/data/shared/blakeo/LM-Reasoning/Reasoning/split_benchmarks/GSM8K_FULL/TEMH"
    os.makedirs(output_dir, exist_ok=True)

    print("Saving new splits...")
    save_jsonl(easy, os.path.join(output_dir, "easy.jsonl"))
    save_jsonl(medium, os.path.join(output_dir, "medium.jsonl"))
    save_jsonl(hard, os.path.join(output_dir, "hard.jsonl"))

    easy_counts = count_sources(easy)
    medium_counts = count_sources(medium)
    hard_counts = count_sources(hard)

    report_lines = [
        "Dataset Distribution by Difficulty Split:\n",
        "Split      | GSM8K | AQUA | OTHER\n",
        "-----------|-------|------|--------",
        f"Easy       | {easy_counts['gsm8k']:5} | {easy_counts['aqua']:4} | {easy_counts['other']:7}",
        f"Medium     | {medium_counts['gsm8k']:5} | {medium_counts['aqua']:4} | {medium_counts['other']:7}",
        f"Hard       | {hard_counts['gsm8k']:5} | {hard_counts['aqua']:4} | {hard_counts['other']:7}",
    ]

    report_path = os.path.join(output_dir, "split_distribution.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print("Done.")

if __name__ == "__main__":
    main()