import json
import os
import re
import torch as th
import pandas as pd
from collections import Counter
import sympy
from prompts import *
from typing import Optional, Union
from reasoners.algorithm import BeamSearchResult

def retrieve_answer(output: Union[list, str]) -> Optional[str]:
    '''
    output should be a world_model.GSM8kState if being a list
    '''
    if output is None:
        return None
    final_answer = {}
    if isinstance(output, list):
        output = output[-1].sub_answer
    # First extract the answer
    match = re.match(r'.*[Tt]he answer is .*?([ $.0-9,\-]+).*\..*', output, re.DOTALL)
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
    final_answer['answer'] = int(answer)
    final_answer['bins'] = bins
    print('final answer: ',final_answer)
    return final_answer

def parse_output(output: BeamSearchResult):
    print('output: ', output)
    if output.terminal_state is None:
        return None
    return retrieve_answer(output.terminal_state.output)
    

def test_output(question: str, output: str):
    if output is None or '=' not in output:
        return False
    if output.split('=')[1].strip() != '24':
        return False
    expression = output.split('=')[0]
    numbers = re.findall(r'\d+', expression)
    question_numbers = re.findall(r'\d+', question)
    if sorted(numbers) != sorted(question_numbers):
        return False
    try:
        return abs(float(sympy.simplify(expression)) - 24) < 1e-6
    except ValueError:
        return False


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


def correct_left_numbers(x: str, y: str, action: str) -> str:
    ## find the actual left numbers
    original_nums = get_current_numbers(y if y else x).split(' ')
    # print(original_nums)
    original_cnt = Counter(original_nums)
    action = action.strip().lower().split(' (left')[0]
    if ' = ' in action:
        expression, new_num = action.split(' = ')[0], action.strip().lower().split(' = ')[1]
        used_nums = re.findall(r'\d+', expression)
        left_nums = [new_num]
        for num in used_nums:
            if num in original_cnt:
                original_cnt[num] -= 1
        for num in original_cnt:
            if original_cnt[num] > 0:
                for _ in range(original_cnt[num]):
                    left_nums.append(num)
    else:
        print(f'no equation in action: {action}')
        left_nums = re.findall(r'\d+', action)
    correct_action = action + ' (left: ' + ' '.join(left_nums) + ')'
    return correct_action


def propose_prompt_wrap(x: str, y: str = '', all_prompt: dict = {}) -> str:
    current_numbers = get_current_numbers(y if y else x)
    if current_numbers == '24':
        # prompt = all_prompt['cot_prompt'].format(input=x) + 'Steps:\n' + y
        prompt = output_prompt.format(input=x) + 'Steps:' + y
        # print(f"Final propose: {prompt}")
    else:
        # prompt = all_prompt['propose_prompt'].format(input=current_numbers)
        prompt = propose_prompt.format(input=current_numbers)
    return prompt


def value_prompt_wrap(x: str, y: str, all_prompt: dict = {}) -> str:
    last_line = y.strip().split('\n')[-1]
    if 'left: ' not in last_line and last_line != '':  # last step
        ans = last_line.lower().replace('answer: ', '')
        # print([value_last_step_prompt.format(input=x, answer=ans)])
        # return all_prompt['value_last_step_prompt'].format(input=x, answer=ans)
        return value_last_step_prompt.format(input=x, answer=ans)
    current_numbers = get_current_numbers(y)
    # return all_prompt['value_prompt'].format(input=current_numbers)
    return value_prompt.format(input=current_numbers)


def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
    if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
        print("not an answer at step 4")
        return 0
    value_names = [_.split('\n')[-1] for _ in value_outputs]
    value_map = {'impossible': 0.001, 'sure': 20}  # TODO: ad hoc
    value = sum(value * value_names.count(name) for name, value in value_map.items())
    return value


def parse_action(input_string):

    # Using re.match to extract both the first list and the list after "left:"
    match_result = re.match(r'.*?(\[.*?\]).*?left: (\[.*?\])', input_string)

    # Extract both lists if the match is found
    first_list = eval(match_result.group(1)) if match_result else None
    left_list = eval(match_result.group(2)) if match_result else None

    return first_list, left_list
