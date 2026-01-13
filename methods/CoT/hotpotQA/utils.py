import re
import string
from typing import Optional
from collections import Counter


def retrieve_answer(output: str) -> Optional[str]:
    match = re.match(r".*[Tt]he answer is (.+?)\..*", output, re.DOTALL)
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
