from typing import Literal, Optional
from reasoners import LanguageModel
# from prompts.game24.prompts import standard_prompt
from datetime import datetime
# from tqdm import tqdm
# import numpy as np
#  import transformers
from collections import Counter
from reasoners.benchmark import Game24Evaluator
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.hf_model import HFModel
from reasoners.lm.llama_api_model import LLaMaApiModel
# import os
# import sys
import json
import fire

def self_consistency(outputs: list[str]):
    outputs = [output.split('=')[0].strip() for output in outputs]
    output_counts = Counter(outputs)
    most_common = output_counts.most_common(1)
    if most_common:
        return [most_common[0][0]]
    else:
        return None 
      
class CoTReasoner:
    def __init__(self, base_model, n_sc=1, temperature=0, bs=1):
        assert (
            n_sc == 1 or temperature > 0
        ), "Temperature = 0 indicates greedy decoding. There is no point running multiple chains (n_sc > 1)"
        self.base_model = base_model
        self.temperature = temperature
        self.n_sc = n_sc
        self.bs = bs

    def __call__(self, example, prompt:dict =None):
        inputs = prompt["cot"].replace("{input}", example)
        outputs = []
        outputs = self.base_model.generate([inputs], 
                                           eos_token_id='\n', 
                                           num_return_sequences=self.n_sc, 
                                           temperature=self.temperature,
                                           do_sample=False, 
                                           additional_prompt='CONTINUE').text
        outputs = [o.replace('Answer:', '').strip() for o in outputs]
        return outputs
       
  

if __name__ == '__main__':

    def main(base_lm: Literal[ 'hf',  'openai', 'api'] = 'openai',
             hf_path: str = 'meta-llama/Llama-2-13b-hf',
             hf_peft_path: Optional[str] = None,
             hf_quantized: Optional[Literal['awq', 'int8', 'fp4', 'nf4']] = None,
             hf_load_awq_path: Optional[str] = None,
             batch_size: int = 1,
             openai_model: str = 'gpt-4o-mini',
             api_model_id='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
             sc_num: int =  1,
             temperature: float = 0.8,
             disable_log: bool = False,
             prompts_file: str = 'prompts/game24/prompts.json',
             resume=0,
             **kwargs):

        if base_lm == 'hf':
            base_model = HFModel(hf_path, hf_path, max_batch_size=batch_size, max_new_tokens=512,
                                 peft_pth=hf_peft_path, quantized=hf_quantized, load_awq_pth=hf_load_awq_path)
        elif base_lm == 'openai':
            base_model = OpenAIModel(openai_model)
        elif base_lm == 'api':
            base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None)
            model_dir = base_model.model_id
        else:
            assert False, f'cannot resolve {base_lm=}'
            
        if base_lm == 'hf' or base_lm == 'api':
            model_name= model_dir.split('/')[-1]
        else:
            model_name = base_lm
            
        with open(prompts_file) as f:
            prompts = json.load(f)
            
        log_dir =  f'logs/game24/'\
                        f'CoT/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        log_dir = log_dir + f'_{model_name}'
        reasoner = CoTReasoner(
            base_model, temperature=temperature, n_sc=sc_num, bs=batch_size
        )
        evaluator = Game24Evaluator(disable_log=disable_log, 
                                  output_extractor=lambda x: self_consistency(x),
                                  answer_extractor=lambda x: (24.0, x),
                                  input_processor=lambda x: x,
                                  prompt=prompts,
                                  sample_prompt_type="cot")
        metric = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=5, resume=resume, log_dir=log_dir)
        print(f'Acc: {metric}')
    


    fire.Fire(main)
