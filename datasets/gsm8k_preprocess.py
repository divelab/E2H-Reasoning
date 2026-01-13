import os
from datasets import load_dataset, disable_caching
disable_caching()


def extract_answer(text):
    text = text.replace(",", "")
    marker_pos = text.find('####')
    answer_text = text[marker_pos + 4:].strip()
    answer_text = answer_text.split()[0]
    return answer_text


def process_row(row: dict) -> dict:
    answer = extract_answer(row["answer"])
    if answer is None:
        print(answer)
    return {
        "question": row["question"],
        "solution": row["answer"],
        "answer": answer,
        "difficulty": row["difficulty_score"],
    }


def main():
    read_path = 'datasets/gsm8k'
    save_path = 'datasets/gsm8k'

    for task in os.listdir(read_path):
        dataset_allsplit = load_dataset("json", data_dir=os.path.join(read_path, task))
        for split in ['train', 'test']:
            dataset = dataset_allsplit[split]
            dataset = dataset.map(process_row, remove_columns=dataset.column_names)
            dataset.to_json(os.path.join(save_path, task, split+".jsonl"))


if __name__ == '__main__':
    main()