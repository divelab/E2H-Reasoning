import re
import pandas as pd
from typing import Optional

"""
Input (x)   : a string of 4 numbers
Output (y)  : a trajectory of 3 steps to reach 24
Reward (r)  : 0 or 1, depending on whether the trajectory is correct
Input Example: 
    1 2 3 4
Output Example: 
    1 + 2 = 3 (left: 3 3 4)
    3 + 3 = 6 (left: 4 6)
    6 * 4 = 24 (left: 24)
    (1 + 2 + 3) * 4 = 24
"""


def read_data(file="24.csv"):
    """
    file: a csv file (fixed)
    """
    data = list(pd.read_csv(file)["Puzzles"])
    return data


def get_input(self, idx: int) -> str:
    return self.data[idx]


def retrieve_answer_from_dataset(example) -> str:
    print("Example", example)
    return (24, str(example))


def parse_response(output: str) -> Optional[str]:
    pattern = r"[Tt]he answer is:\s*(.+)"

    match = re.search(pattern, output)

    if match is None:
        return None

    answer = match.group(1).strip()
    return [answer]
