import os
from datasets import load_dataset, concatenate_datasets, disable_caching
disable_caching()


def last_boxed_only_string(string: str) -> str:
    
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
        
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def process_row(row: dict) -> dict:
    answer = remove_boxed(last_boxed_only_string(row["solution"]))
    return {
        "question": row["problem"],
        "solution": row["solution"],
        "answer": answer,
        "type": row["type"],
        "level": row["level"]
    }


def main():
    read_path = 'EleutherAI/hendrycks_math'
    save_path = 'datasets/math'

    dataset_types = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

    levels = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']

    for split in ['train', 'test']:

        dataset = []
        for dataset_type in dataset_types:
            dataset.append(load_dataset(path=read_path, name=dataset_type, split=split))
        dataset = concatenate_datasets(dataset)

        dataset = dataset.map(process_row, remove_columns=dataset.column_names)

        for level in levels:
            level_dataset = dataset.filter(lambda row: row['level']==level)
            level_dataset.to_json(os.path.join(save_path, level.lower().replace(" ", ""), split+".jsonl"))



if __name__ == '__main__':
    main()