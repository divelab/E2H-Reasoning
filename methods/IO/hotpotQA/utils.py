import re
import string
from collections import Counter
from typing import Optional


def retrieve_answer(output: str) -> Optional[str]:
    match = re.match(r".*[Tt]he answer is (.+?)\..*", output, re.DOTALL)
    # Might break
    if match is None:
        return None
    answer = match[1]
    return answer


def retrieve_answer_from_dataset(answer: str) -> str:
    return answer


def judge_answer(output: Optional[str], answer: str) -> bool:
    if output is None:
        return False
    try:
        output = int(output)
        answer = int(answer)
        return output == answer
    except ValueError:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output == answer
    except ValueError:
        pass
    return output == answer


def rap_extractor(algo_output, aggregate=True):
    from reasoners.algorithm import MCTSAggregation

    if aggregate:
        aggregator = MCTSAggregation(
            retrieve_answer, weight_policy="edge_inverse_depth"
        )
        output = aggregator(algo_output.tree_state)
    else:
        if algo_output.terminal_state is None:
            output = None
        else:
            output = retrieve_answer(algo_output.terminal_state)
    return output


def cot_basic_extractor(algo_output):
    answer = retrieve_answer(algo_output)
    return answer


# You have a list of strings with len of 1
def cot_hotpot_basic_extractor(algo_output):
    answer = retrieve_answer(algo_output[0])
    return answer


def cot_sc_extractor(algo_output, sc=True):
    # aggregate the results from multiple reasoning chains with majority vote
    answers = [retrieve_answer(x) for x in algo_output]
    answers = [x for x in answers if x is not None]
    counter = Counter(answers)
    if counter == {}:
        return None
    return counter.most_common(1)[0][0]


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
