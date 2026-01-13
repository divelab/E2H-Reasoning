import re
from typing import Optional
from collections import Counter
import numpy as np


def retrieve_answer(output: str) -> Optional[str]:
    match = re.match(r".*[Tt]he answer is:?\s*(.+?)\..*", output, re.DOTALL)
    # Might break
    if match is None:
        if (
            output is not None
        ):  # many times the answer is not in the format "The answer is: ..."
            return output.strip(".").strip()
        return None
    answer = match[1]
    return answer


def cot_sc_extractor(algo_output, sc=True):
    answers = [retrieve_answer(x) for x in algo_output]
    answers = [x for x in answers if x is not None]
    counter = Counter(answers)
    if counter == {}:
        return None
    return counter.most_common(1)[0][0]


def retrieve_answer_from_dataset(example):
    state_numbers = range(1, 25)
    output = np.array([])
    for state_number in state_numbers:
        state_str = f"state_{state_number}"
        num = example[state_str]
        output = np.append(output, num)
    return example["moves"], output
