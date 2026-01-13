import json
import os
import argparse

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(line.strip()) for line in f]

def save_jsonl(data, filename):
    with open(filename, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def normalize_keys(data):
    normalized_data = []
    for item in data:
        if "correct" in item:
            item["answer"] = item.pop("correct")
        normalized_data.append(item)
    return normalized_data

def normalize_directory(directory):
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    total_files = 0
    total_items_normalized = 0
    
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(directory, filename)
            data = load_jsonl(file_path)
            correct_count = sum(1 for item in data if "correct" in item)
            normalized_data = normalize_keys(data)
            save_jsonl(normalized_data, file_path)
            
            print(f"Processed {file_path}: renamed {correct_count} keys")
            total_files += 1
            total_items_normalized += correct_count
    
    print(f"Normalized {total_items_normalized} keys across {total_files} files")

def main():
    parser = argparse.ArgumentParser(description="Normalize keys in JSONL files")
    parser.add_argument("directory", help="Directory containing JSONL files")
    args = parser.parse_args()
    normalize_directory(args.directory)

if __name__ == "__main__":
    main()