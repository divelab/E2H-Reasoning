import json
import random
from reasoners import Evaluator

class StrategyQAEvaluator(Evaluator):
    def __init__(self,
                 output_extractor,
                 answer_extractor,
                 init_prompt=None,
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="l2m",
                 resume = 0,
                 dataset_limit = 200,
                 dataset_path = 'data/strategyqa/strategyqa_test.json') -> None:

        super().__init__()
        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.input_processor = lambda x: x["question"]
        self._dataset_name = 'strategyQA'
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.sample_prompt_type = sample_prompt_type
        with open(dataset_path, 'r') as f:
          self.full_dataset = json.load(f)[resume:dataset_limit]

    def sample_prompt(self,
                      shuffle_prompt=True,
                      num_shot=4):

        sample_prompt_type = self.sample_prompt_type
        if sample_prompt_type == "cot" or sample_prompt_type == "tot":
            prompt = {}
            key = 'cot_pool'
            if shuffle_prompt:
                examples = random.sample(self.init_prompt[key], num_shot)
            else:
                examples = self.init_prompt[key][:num_shot]
            prompt[sample_prompt_type] = "\n\n".join(examples)
            if sample_prompt_type == "tot":
                prompt[sample_prompt_type] = prompt[sample_prompt_type] + self.init_prompt["prefix"]
                prompt["tot_prefix"] = self.init_prompt["tot_prefix"]
                prompt["prefix"] = self.init_prompt["prefix"]
                prompt["self-eval"] = self.init_prompt["self-eval"]
                prompt['rating_prompt'] = self.init_prompt['rating_prompt']
                prompt['propose_action_instructions'] = self.init_prompt['propose_action_instructions']
                prompt['propose_answer_instructions'] = self.init_prompt['propose_answer_instructions']
        elif sample_prompt_type == "o1":
            prompt = {}
            instructions = self.init_prompt["o1_instructions"]
            prompt["o1"] = self.init_prompt["cot_prefix"] + instructions
        elif sample_prompt_type == "rap":
            return None
        else:
            raise NotImplementedError
        return prompt

    def eval_output(self, answer, output):
        if output is None:
            return False
        
        # False vs no and True vs yes
        answer = "no" if not answer else "yes"
        
        return answer == output.strip().lower()