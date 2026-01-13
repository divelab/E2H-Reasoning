import re
import string
from collections import Counter
from typing import Optional

value_map = {"sure": 1, "likely": 0.1, "impossible": 0.0001}


def retrieve_answer(output: str) -> Optional[str]:
    match = re.match(r".*[Tt]he answer is (.+?)\..*", output, re.DOTALL)
    # Might break
    if match is None:
        return None
    answer = match[1]
    return answer


def retrieve_answer_from_dataset(answer: str) -> str:
    # print(f"Answer type: {type(answer)}")
    # print(f"Answer content: {answer}")
    return answer


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def tot_extractor(algo_output):
    ans = ""
    try:
        ans = retrieve_answer(algo_output.terminal_state.last_action)
    #   print(f"Answer: {ans}")
    #   print(f"Output: {algo_output.terminal_state.last_action}")
    except Exception:
        return None

    return ans
