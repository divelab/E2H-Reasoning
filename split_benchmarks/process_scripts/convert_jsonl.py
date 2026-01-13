import json
import argparse

def convert_json_to_jsonl(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Write as JSONL (one JSON object per line)
    with open(output_file, 'w') as f:
        # If the input is a list/array
        if isinstance(data, list):
            for item in data:
                f.write(json.dumps(item) + '\n')
        # If the input is a dictionary/object
        else:
            f.write(json.dumps(data) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Convert JSON to JSONL")
    parser.add_argument("input", help="Input JSON file")
    parser.add_argument("output", help="Output JSONL file")
    args = parser.parse_args()
    
    convert_json_to_jsonl(args.input, args.output)
    print(f"Converted {args.input} to {args.output}")

if __name__ == "__main__":
    main()