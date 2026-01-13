
from typing import Optional, Literal
import os
from datetime import datetime
from reasoners import LanguageModel, Reasoner
from reasoners import LanguageModel 
from reasoners.algorithm import HeuristicGuidedSearch
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.llama_api_model import LLaMaApiModel

from utils import create_callable_function, CubeEvaluator

from world_model import CubeWorldModel
from search_config import CubeConfig
import fire
import json

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

def autohd_search(base_model: LanguageModel,
           prompt: dict,
           data_path: str = './data/cube_test.csv',
           resume: int = 0,
           depth_limit: int = 6,
           log_dir: Optional[str] = None,
           disable_log: bool = False,
           temperature: float = 1.0,
           heuristic_fn: callable = None,
           add_cost: bool = True,
           beam_size = 10,
           **search_algo_params):

    search_algo_params |= {"max_depth": depth_limit}
    
    if heuristic_fn is not None:
        log_dir = f'logs/Heuristic_search/rubikscube/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    
    world_model = CubeWorldModel(base_model=base_model, prompt=prompt, max_steps=depth_limit)
    config = CubeConfig(base_model=base_model, prompt=prompt, temperature=temperature, heuristic_fn=heuristic_fn)
    
    search_algo = HeuristicGuidedSearch(beam_size = beam_size, add_cost = add_cost, **search_algo_params)
    
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    evaluator = CubeEvaluator(data_path=data_path, init_prompt=prompt, disable_log=disable_log)
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print("Final ACC: ", accuracy)
    return accuracy


def main(
        base_lm: Literal['openai', 'api']  = 'openai',
        data_path: str = 'data/cube/cube_test.csv',
        disable_log: bool = False,
        depth_limit: int = 5,
        temperature = 1.0,
        action_prompt = False,
        n_iter: int = 3,
        beam_size = 10,
        add_cost:bool = True,
        prompt_path = "prompts/cube/prompts.json",
        api_model_id='meta-llama/Meta-Llama-3.1-70B-Instruct',
        openai_model='gpt-4o-mini',
        heuristic_log_file = '',
        **kwargs
        ):
    
    with open(prompt_path) as f:
        prompt = json.load(f)

    if base_lm == "openai":
        base_model = OpenAIModel(openai_model)
        model_name = f'{base_lm}_{openai_model}'
    elif base_lm == 'api':
        base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None, additional_prompt="CONTINUE")
        model_name = f'{base_lm}_{api_model_id.replace("/", "-")}'
    else:
        raise NotImplementedError

    # Load testset with initial config, otherwise prompt model to generate it first.     
    log_dir =  f'logs/rubikscube/autoHD/{model_name}/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    
    # logs storage
    heuristic_fn = None
    # print(os.path.exists(heuristic_log_file))
    if os.path.exists(heuristic_log_file):
        with open(heuristic_log_file) as f:
            heuristic_log = f.read()
        heuristic_fn = load_heuristic_fn(heuristic_log)

    autohd_search(base_model,
            prompt,
            disable_log=disable_log,
            data_path=data_path,
            depth_limit=depth_limit,
            action_prompt = action_prompt,
            temperature=temperature,
            n_iters = n_iter, 
            add_cost = add_cost, 
            beam_size = beam_size,
            heuristic_fn=heuristic_fn,
            log_dir = log_dir,
            **kwargs)

if __name__ == '__main__':
    fire.Fire(main)

        
        