import re
from typing import Optional, Union
from collections import Counter

def retrieve_answer(output: Union[list, str]) -> Optional[str]:
    '''
    output should be a world_model.GSM8kState if being a list
    '''
    if isinstance(output, list):
        output = output[-1].sub_answer
    # print("output:", output)
    match = re.match(r'.*[Tt]he answer is .*?([ $.0-9,\-]+).*\..*', output, re.DOTALL)
    if match is None:
        return None
    answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    # print("answer:", answer)
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    return answer


def retrieve_answer_from_dataset(answer: str) -> str:
    return re.match(r'[\S\s]*#### (.*)$', answer)[1]


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
        aggregator = MCTSAggregation(retrieve_answer, weight_policy='edge_inverse_depth')
        output = aggregator(algo_output.tree_state)
    else:
        if algo_output.terminal_state is None:
            output = None
        else:
            output = retrieve_answer(algo_output.terminal_state)
    return output

def cot_sc_extractor(algo_output, sc=True):
    # aggregate the results from multiple reasoning chains with majority vote
    answers = [retrieve_answer(x) for x in algo_output]
    answers = [x for x in answers if x is not None]
    counter = Counter(answers)
    if counter == {}:
        return None
    return counter.most_common(1)[0][0]

def tot_extractor(algo_output):
    ans = ""
    try:
      ans = algo_output.terminal_state.last_action
      print("Ans: ", ans)
    except:
      return None
    ans = ans.replace(',', '').replace('$', '')

    # Check for the presence of '=' and extract the number after it
    matches_equals = re.findall(r'=\s*(\d+)', ans)
    if matches_equals:
        # Return the last number after the last '='
        return int(matches_equals[-1])

    # If no '=' is found, extract the first number
    match_number = re.search(r'\d+', ans)
    if match_number:
        return int(match_number.group())
    
    return None