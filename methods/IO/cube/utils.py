import re

import numpy as np


def retrieve_answer(output: str):
    match = re.match(r".*[Tt]he answer is\s*(.+)", output, re.DOTALL)
    if match is None:
        return None
    answer = match[1].strip()
    return answer


def retrieve_answer_from_dataset(example):
    state_numbers = range(1, 25)
    output = np.array([])
    for state_number in state_numbers:
        state_str = f"state_{state_number}"
        num = example[state_str]
        output = np.append(output, num)
    return example["moves"], output
