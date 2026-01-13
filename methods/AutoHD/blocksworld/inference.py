from typing import Type, Callable, Optional, Literal

import numpy as np
from reasoners import LanguageModel, Reasoner
from reasoners.benchmark import BWEvaluator
from reasoners.algorithm import HeuristicGuidedSearch
from reasoners.lm import  HFModel, OpenAIModel, LLaMaApiModel
from world_model import BlocksWorldModel, BWState, BWAction
from search_config import BWConfig
import json
import fire
import os
from action_generation_prompts import get_next_actions_empty, get_next_actions_holding
from utils import create_callable_function

heuristic_strs = {
    "best": "Test the best heuristic in all the generations: ",
    "last": "Test the best heuristic in the last generation: "
}

def load_heuristic_fn(log_content, heuristic_fn_type: Literal['best', 'last'] = 'best'):
    lines = log_content.split('\n')
    start_idx = None
    for idx, line in enumerate(lines):
        if heuristic_strs[heuristic_fn_type] in line:
            start_idx = idx +1
            break
    assert start_idx is not None, f"Cannot find the heuristic function in the log file."
    extracted_code = "\n".join(lines[start_idx:]).replace('Function:', '').strip()
    heuristic_fn = create_callable_function(extracted_code)
    return heuristic_fn


def plan_extractor(algo_output):
    try:
        return "\n".join(algo_output[0].terminal_node.state.action_history)
    except Exception as e:
        print("Error in output extraction,", e)
        return ""
    
def autohd_search(base_model: LanguageModel,
           prompt: dict,
           data_path: str = 'data',
           resume: int = 0,
           depth_limit: int = 6,
           log_dir: Optional[str] = None,
           disable_log: bool = False,
           domain_file: str = "",
           config_file: str = "",
           lm_plan_file: str = 'lm_plan.tmp',
           step_into_state: bool = False,
           temperature: float = 0.8,
           action_prompt: bool = False,
           heuristic_fn: callable = None,
           **search_algo_params):

    search_algo_params |= {"max_depth": depth_limit}
    
    world_model = BlocksWorldModel(base_model=base_model, prompt=prompt, max_steps=depth_limit)
    config = BWConfig(base_model=base_model, prompt=prompt, temperature=temperature, step_into_state=step_into_state, action_prompt = action_prompt, heuristic_fn=heuristic_fn)
    
    search_algo = HeuristicGuidedSearch(add_cost=False, **search_algo_params)
    
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    evaluator = BWEvaluator(config_file=config_file, 
                            domain_file=domain_file, 
                            data_path=data_path, 
                            init_prompt=prompt, 
                            disable_log=disable_log, 
                            output_extractor=plan_extractor)
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(accuracy)
    return accuracy


def main(
            base_lm: Literal['openai', 'together-ai', 'llamaapi']  = 'openai',
            prompt_path: str = 'prompts/blocksworld/pool_prompt_v1.json',
            disable_log: bool = False,
            config_file: str = "data/blocksworld/bw_config.yaml",
            domain_file: str = "data/blocksworld/generated_domain.pddl",
            lm_plan_file: str = 'lm_plan.tmp',
            temperature = 0.8,
            step_into_state = True,
            action_prompt = True,
            n_steps: int = 2,
            n_iters:int = 10, 
            api_model_id = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
            openai_model = 'gpt-4o-mini',
            heuristic_log_file = None,
            **kwargs
            ):
    
        data_path = f'data/blocksworld/split_v1/split_v1_step_{n_steps}_data.json'
        with open(prompt_path) as f:
            prompt = json.load(f)
        
        if action_prompt:
            prompt['next_actions_holding'] = get_next_actions_holding(prompt)
            prompt['next_actions_empty'] = get_next_actions_empty(prompt)

        if base_lm == "openai":
            base_model = OpenAIModel(openai_model, additional_prompt="CONTINUE")
        elif base_lm == "llamaapi":
            base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None, additional_prompt="CONTINUE")
        elif base_lm == "together-ai":
            base_model = HFModel(None, None, additional_prompt = "CONTINUE")
        else:
            raise NotImplementedError
        
        if os.path.exists(heuristic_log_file):
            with open(heuristic_log_file) as f:
                heuristic_log = f.read()
        heuristic_fn = load_heuristic_fn(heuristic_log)
        
        autohd_search(base_model,
               prompt,
               disable_log=disable_log,
               data_path=data_path,
               config_file=config_file,
               depth_limit=n_steps,
               domain_file=domain_file,
               action_prompt = action_prompt,
               lm_plan_file=lm_plan_file,
               step_into_state = step_into_state,
               temperature=temperature, 
               heuristic_fn=heuristic_fn,
               n_iters = n_iters, **kwargs)


if __name__ == '__main__':
    fire.Fire(main) # user will need to switch the model in the code
