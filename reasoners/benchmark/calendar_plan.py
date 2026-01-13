import copy
import random

import datasets

from reasoners import Evaluator

SEED = 42

class CalendarPlanEvaluator(Evaluator):
    def __init__(self,
                 output_extractor,
                 answer_extractor,
                 init_prompt=None,
                 disable_log=False,
                 dataset_path=None,
                 disable_tqdm=False,
                 sample_prompt_type="l2m",
                 prompt_key="prompt_5shot") -> None:
        super().__init__()
        self.parsed_plan = None
        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        def input_processor_debug(x):
            input = x[prompt_key]
            return input

        self.input_processor = input_processor_debug

        full_dataset = datasets.load_dataset("json", data_files=dataset_path, split="train")
        shuffled_dataset = full_dataset.shuffle(seed=SEED)
        selected_subset = shuffled_dataset.select(range(20))
        self.full_dataset = selected_subset
        self._dataset_name = "calendar_plan"
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.sample_prompt_type = sample_prompt_type
        
    def sample_prompt(self, shuffle_prompt=True, num_shot=4):
        sample_prompt_type = self.sample_prompt_type
        if sample_prompt_type == "l2m":
            prompt = {}
            if shuffle_prompt:
                decomp_examples = random.sample(
                    self.init_prompt["decomposition_pool"], num_shot
                )
                solv_examples = random.sample(
                    self.init_prompt["solving_pool"], num_shot
                )
            else:
                decomp_examples = self.init_prompt["decomposition_pool"][:num_shot]
                solv_examples = self.init_prompt["solving_pool"][:num_shot]
            prompt["decomposition"] = (
                "".join(decomp_examples) + self.init_prompt["composition_prefix"]
            )
            prompt["overall"] = (
                "".join(decomp_examples) + self.init_prompt["overall_prefix"]
            )
            prompt["solving"] = (
                "".join(solv_examples) + self.init_prompt["solving_prefix"]
            )

        elif sample_prompt_type == "cot" or sample_prompt_type == "tasb":
            prompt = {}
            prompt[sample_prompt_type] = "{Question}"
        elif sample_prompt_type == "tot":
            prompt = self.init_prompt
            prompt["cot_prefix"] = self.init_prompt["cot_prefix"]
            prompt[sample_prompt_type] = self.init_prompt["prefix"]
        elif sample_prompt_type == "rap":
            ret = copy.deepcopy(self.init_prompt)
            ret["interactive_examples"], ret["useful_examples"] = zip(
                *random.sample(
                    list(zip(ret["interactive_examples"], ret["useful_examples"])),
                    k=num_shot,
                )
            )
            return ret
        elif sample_prompt_type == "grace":
            return None
        elif sample_prompt_type == "o1":
            prompt = {}
            prompt[sample_prompt_type] = self.init_prompt["o1_instructions"] + "\n<Question>\n{Question}\n</Question>"
        else:
            raise NotImplementedError
        return prompt
    
    def eval_output(self, answer, output):
        if output is None or output[0] == '':
            return 0.0
        r_day, r_start_hour, r_end_hour = answer
        s_day, s_start_hour, s_end_hour = output
        if r_day == s_day and r_start_hour == s_start_hour and r_end_hour == s_end_hour:
            return 1.0
        return 0.0