import json
import re
import argparse
from difflib import get_close_matches

def load_questions(file_path, label):
    """
    Load questions from a JSONL file and tag each with a difficulty label.
    Returns a dict mapping question text to its difficulty label.
    """
    q_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            q = data.get("question", "").strip()
            if q:
                q_map[q] = label
    return q_map


def extract_question(prompt):
    """
    Extract the math problem question from the model prompt text.
    Assumes the prompt contains a line like:
      "Solve the following math problem\n<QUESTION>\n\n Show your work..."
    """
    # Regex to capture between 'Solve the following math problem' and the blank line before 'Show your work'
    pattern = r"Solve the following math problem\s*(.+?)\s*\n\n"
    m = re.search(pattern, prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: simple split
    if "Solve the following math problem" in prompt and "Show your work" in prompt:
        part = prompt.split("Solve the following math problem", 1)[1]
        question = part.split("Show your work", 1)[0]
        return question.strip()
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Map evaluated prompts to their difficulty based on question files."
    )
    parser.add_argument("--eval", required=True, help="Path to evaluation JSON file")
    parser.add_argument("--output", required=False, default="accuracy_by_difficulty.txt"),
    args = parser.parse_args()
    
    trivial = "/mnt/data/shared/blakeo/LM-Reasoning/Reasoning/split_benchmarks/GSM8K_FULL/TEMH/trivial.jsonl"
    easy = "/mnt/data/shared/blakeo/LM-Reasoning/Reasoning/split_benchmarks/GSM8K_FULL/TEMH/easy.jsonl"
    medium = "/mnt/data/shared/blakeo/LM-Reasoning/Reasoning/split_benchmarks/GSM8K_FULL/TEMH/medium.jsonl"
    hard = "/mnt/data/shared/blakeo/LM-Reasoning/Reasoning/split_benchmarks/GSM8K_FULL/TEMH/hard.jsonl"

    # Build a map of question -> difficulty
    q_map = {}
    q_map.update(load_questions(trivial, "trivial"))
    q_map.update(load_questions(easy, "easy"))
    q_map.update(load_questions(medium, "medium"))
    q_map.update(load_questions(hard, "hard"))
    

    difficulties = ["trivial", "easy", "medium", "hard", "unknown"]
    counts = {d: 0 for d in difficulties}
    correct = {d: 0.0 for d in difficulties}

    # Load evaluation results
    with open(args.eval, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    print(len(eval_data.get("detailed_results", [])))

    # Process each detailed result
    for idx, res in enumerate(eval_data.get("detailed_results", []), 1):
        prompt = res.get("prompt", "")
        question = extract_question(prompt)
        if not question:
            print("Could not extract question")
        else:
            difficulty = q_map.get(question)
            if not difficulty:
                print("could not map question to difficulty")
        score = res.get("score", 0.0)
        counts[difficulty] = counts.get(difficulty, 0) + 1
        if (score > 0.5):  
          correct[difficulty] = correct.get(difficulty, 0.0) + 1

        # Prepare accuracy results
    lines = []
    lines.append("Accuracy by difficulty:")
    for d in difficulties:
        total = counts[d]
        if total > 0:
            acc = correct[d] / total
            line = f"{d.capitalize():6s}: {acc:.2%} ({correct[d]:.1f}/{total})"
        else:
            line = f"{d.capitalize():6s}: no examples"
        lines.append(line)
    output_text = "\n".join(lines)

    with open(args.output, 'w', encoding='utf-8') as out_f:
        out_f.write(output_text)

    print(output_text)

if __name__ == "__main__":
    main()
