import re
from typing import Optional, Union
from reasoners.base import AlgorithmOutput
from collections import Counter
import ast


def retrieve_answer(output: str) -> Optional[str]:
    '''
    output should be a world_model.GSM8kState if being a list
    '''
    final_answer = {}
    # First extract the answer
    match = re.search(r'[Tt]he answer is\s*([$\d.,-]+)', output, re.DOTALL)

    if match is None:
        answer = None
    else:
        answer = match.group(1).replace(',', '').replace('$', '').replace(' ', '').replace('.', '')

    bins_matches = re.findall(r'\[([^\]]+)\]', output)
    bins = []
    for match in bins_matches:
        try:
            bin_list = ast.literal_eval(f'[{match}]')
            bins.append(bin_list)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing bin '{match}': {e}")
            bins = None
            break

    try:
        final_answer['answer'] = int(answer)
    except Exception as e:
        print(f"Error parsing answer '{answer}': {e}")
        return None
    final_answer['bins'] = bins
    # print("Final answer:", final_answer)
    return final_answer