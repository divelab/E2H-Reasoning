from typing import Any, List
from reasoners import Evaluator
import pandas as pd
import re

def read_data(data_path="data/game24/24.csv"):
    """
    file: a csv file (fixed)
    """
    if data_path is None:
        data_path = "data/game24/24.csv"
    data = list(pd.read_csv(data_path)["Puzzles"])[900:1000]
    return data

def input_processor_debug(x):
    # input = x['prompt_0_shot']
    # o1 input is only the numbers, might have to change for other methods
    input = x
    return input


class Game24Evaluator(Evaluator):
    def __init__(
        self,
        output_extractor,
        answer_extractor,
        disable_log=False,
        disable_tqdm=False,
        input_processor=input_processor_debug,
        prompt = None,
        sample_prompt_type="l2m",
        test_at_n=1,
        heuristic_search=False,
        data_path=None
    ) -> None:
        super().__init__()
        self.parsed_plan = None
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.full_dataset = read_data(data_path=data_path)
        self.sample_prompt_type = sample_prompt_type
        self._dataset_name = "game24"
        self.input_processor = input_processor
        self.heuristic_search = heuristic_search
        self.test_at_n = test_at_n
        self.prompt = prompt

    def sample_prompt(self, shuffle_prompt, num_shot):
        sample_prompt_type = self.sample_prompt_type
        prompt = {}
        if sample_prompt_type == "cot" or sample_prompt_type == "o1":
            prompt[sample_prompt_type] = self.prompt[f'{sample_prompt_type}_prompt']
            # self.prompt[f'{sample_prompt_type}_prompt'] = standard_prompt
        elif sample_prompt_type == "beamstar":
            prompt['action_prompt'] = self.prompt['action_prompt_autohd']
            prompt['output_prompt'] = self.prompt['output_prompt']
            
        elif sample_prompt_type == "rap" or sample_prompt_type == "tot":
            return None
        else:
            return NotImplementedError

        return prompt

    def eval_output(self, answer: tuple, output: List[str] = None) -> bool:
        target, question = answer
        if not output:
            print("Output generated was not of format.")
            return False

        for op in output:
            try:
                evaluated_output = eval(op)
            except Exception as e:
                continue
            print("Output", op)
            numbers = re.findall(r"\d+\.\d+|\d+", op)
            question_numbers = re.findall(r"\d+", question)
            numbers = [float(x) for x in numbers]
            question_numbers = [float(x) for x in question_numbers]
            print("Answer here", answer, evaluated_output)
            if sorted(numbers) != sorted(question_numbers):
                continue
            elif abs(target - evaluated_output) < 1e-6:
                return True
        return False
