
from typing import Any
from reasoners import Evaluator
import random
import csv

def load_dataset():
    csv_path = 'data/binpacking/test.csv'
    data = []
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            row['item_sizes'] = list(map(int, row['item_sizes'].split()))
            row['bin_capacity'] = int(row['bin_capacity'])
            row['optimal_bins'] = int(row['optimal_bins'])
            data.append(row)
    return data

class BinPackingEvaluator(Evaluator):
    
    
    def __init__(self,
                 output_extractor,
                 answer_extractor,
                 init_prompt=None,
                 disable_log=False,
                 disable_tqdm=False,
                 input_processor=lambda x: x,
                 sample_prompt_type="cot",
                 heuristic_search=False) -> None:
        super().__init__()
        self.parsed_plan = None
        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.full_dataset = load_dataset()
        self.sample_prompt_type = sample_prompt_type
        self._dataset_name = 'binpacking'
        self.input_processor = input_processor
        self.heuristic_search = heuristic_search
           
    def sample_prompt(self, shuffle_prompt, num_shot):
      sample_prompt_type = self.sample_prompt_type
      prompt = {}
      if sample_prompt_type == "cot":
          pool_key = f'cot_pool'
          if shuffle_prompt:
              examples = random.sample(self.init_prompt[pool_key], num_shot)
          else:
              examples = self.init_prompt[pool_key][:num_shot]
          prompt[sample_prompt_type] =self.init_prompt["introduction"] + "".join(examples) + self.init_prompt["prefix"]
      elif sample_prompt_type == "tot":
          prompt['output_prompt'] = self.init_prompt['output_prompt']
          prompt['propose_prompt'] = self.init_prompt['propose_prompt']
          prompt['value_prompt'] = self.init_prompt['value_prompt']
          prompt['value_last_step_prompt'] = self.init_prompt['value_last_step_prompt']
      elif sample_prompt_type == "o1":
          prompt[sample_prompt_type] =self.init_prompt["introduction"] + self.init_prompt["o1_instructions"] +  self.init_prompt["prefix"]
      elif sample_prompt_type == "beamsearch":
          return None 
      else:    
          return NotImplementedError
      return prompt
    
    def eval_output(self, answer, output):
        if output is None or output['answer'] is None or output['bins'] is None: # LLM didn't return an answer
            return False
        if answer['optimal_bins'] != output['answer']: # Incorrect number of bins
            return False
        bins = output['bins']
        item_sizes = answer['item_sizes']
        answer_items = []
        for bin in bins:
            bin_weight = sum(bin)
            if bin_weight > answer['bin_capacity']: # Bin exceeds capacity
                return False
            answer_items.extend(bin)
        if sorted(answer_items) != sorted(item_sizes): # Items don't match, LLM hallucinated.
            return False
        return True

  