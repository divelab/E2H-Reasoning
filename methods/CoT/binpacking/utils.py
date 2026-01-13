import re
from typing import Optional, Union
from reasoners.base import AlgorithmOutput
from collections import Counter


def retrieve_answer(output: Union[list, str]) -> Optional[str]:
    '''
    output should be a world_model.GSM8kState if being a list
    '''
    final_answer = {}
    if isinstance(output, list):
        output = output[-1].sub_answer
    # First extract the answer
    match = re.match(r'.*[Tt]he answer is .*?([ $.0-9,\-]+).*,?.*', output, re.DOTALL)
    if match is None:
        return None
    answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    
    # Extract the bins
    bins_match = re.search(r'\[\[.*\]\]', output)
    try:
        bins = eval(bins_match.group()) if bins_match else None
    except:
        return None
    # print("answer:", answer)
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    try:
        final_answer['answer'] = int(answer)
    except:
        return None
    final_answer['bins'] = bins
    print(final_answer)
    return final_answer

def cot_sc_extractor(algo_output, sc=True):
    # aggregate the results from multiple reasoning chains with majority vote
     # aggregate the results from multiple reasoning chains with majority vote
    
    if not algo_output:
        return None
    answers = [retrieve_answer(x) for x in algo_output]
    tuples_list = [
        (entry['answer'], tuple(map(tuple, entry['bins'])))
        for entry in answers
        if entry is not None and 'answer' in entry and entry['answer'] is not None and 'bins' in entry and entry['bins'] is not None
    ]
    
    count = Counter(tuples_list)
    
    most_common = count.most_common(1)
    if most_common and most_common[0][1] > 1:  # Ensure it's not a tie with count > 1
        answer, bins = most_common[0][0]
        return {'answer': answer, 'bins': [list(bin) for bin in bins]}
    
    # Otherwise, return the first non-None element
    for entry in answers:
        if entry is not None:
            return entry
    
    # If all entries are None, return None
    return None