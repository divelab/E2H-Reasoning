"""
GSM8KRewardModel - Helper class for evaluating and scoring GSM8K equations
"""

import re

class GSM8KRewardModel:
    @staticmethod
    def is_correct(extracted_answer, answer):
        answer = float(answer)
        extracted_answer = extracted_answer.replace(",", "")
        extracted_answer = re.search(r'-?\d+(?:\.\d+)?', extracted_answer)
        if extracted_answer is not None:
            extracted_answer = float(extracted_answer.group(0))
            return abs(answer - extracted_answer) < 1e-5
        return False
