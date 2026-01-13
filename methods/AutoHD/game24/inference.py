from datetime import datetime

import json
import fire
import os
from typing import Sequence, Any, Literal, Optional
import json
from reasoners.lm.hf_model import HFModel
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.llama_api_model import LLaMaApiModel
from reasoners import LanguageModel, Reasoner
from reasoners.benchmark import Game24Evaluator
from utils import parse_result, create_callable_function
from reasoners.algorithm import BeamSearch, DFS, HeuristicGuidedSearch
from search_config import Game24Config
from world_model import Game24WorldModel

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
           search_algo: str = "beam",
           data_path: str = 'data',
           resume: int = 0,
           depth_limit: int = 4,
           log_dir: Optional[str] = None,
           disable_log: bool = False,
           temperature: float = 0.8,
           heuristic_fn: callable = None,
           n_iters: int = 1,
           terminal_beam: int = 1,
           n_sc = 1,
           heuristic_search_type='test',
           **search_algo_params):

    if search_algo == "beam":
        search_algo_params |= {"max_depth": depth_limit}
    elif search_algo == "dfs":
        search_algo_params |= {"depth": depth_limit,}
    elif search_algo == "beamstar":
        search_algo_params |= {"max_depth": depth_limit, "n_iters": n_iters, 'terminal_beam_size': terminal_beam}
    else:
        print("Unknown search algorithm", search_algo)
        raise NotImplementedError
    
    if heuristic_fn is not None:
        log_dir = f'logs/Heuristic_search/game24/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    world_model = Game24WorldModel(base_model=base_model, prompt=prompt, n_sc=n_sc)
    config = Game24Config(base_model=base_model, 
                          prompt=prompt, 
                          temperature=temperature, 
                          heuristic_fn=heuristic_fn,
                          heuristic_search_type=heuristic_search_type)
    
    # output_extractor = dfs_bw_extractor if search_algo == "dfs" else bfs_bw_extractor
    if search_algo == "dfs":
        search_algo = DFS(**search_algo_params)
    elif search_algo == "beam":
        search_algo = BeamSearch(**search_algo_params)
    elif search_algo == "beamstar":
        search_algo = HeuristicGuidedSearch(**search_algo_params)
    else:
        raise NotImplementedError
    
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    evaluator = Game24Evaluator(  
                                  prompt=prompt, 
                                  disable_log=disable_log, 
                                  output_extractor=parse_result,
                                  answer_extractor=lambda x: (24.0, x),
                                  input_processor=lambda x: x,
                                  sample_prompt_type="beamstar",
                                  heuristic_search=True,
                                  test_at_n=terminal_beam
                                  )
    metric = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=5, resume=resume, log_dir=log_dir)
    print(metric)
    return metric # When doing Heuristic search, this will return the Levenshtein distance.


def main(base_lm:Literal['hf', 'api', 'openai'] = 'openai',
         model_dir= None,  
         data_path="data/game24/24.csv",
         prompt_path="prompts/game24/prompts.json",  
         quantized='int8',
         resume=0, 
         temperature=0.8,
         sc_num=1,
         log_dir=None,
         depth_limit=4,
         num_solutions=1,
         n_iters=5,
         heuristic_log_file='',
         api_model_id='meta-llama/Meta-Llama-3.1-70B-Instruct',
         openai_model='gpt-4o-mini',
         **kwargs):


    if base_lm == "openai":
        base_model = OpenAIModel(openai_model, additional_prompt="CONTINUE")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir,quantized=quantized)
    elif base_lm == 'api':
        base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None, additional_prompt="CONTINUE")
        model_name = f'{base_lm}_{api_model_id.replace("/", "-")}'
    else:
        raise ValueError(f"base_lm {base_lm} is not supported")
    
    if base_lm == 'hf' or base_lm == 'api':
        model_name = model_dir.split('/')[-1]
    else:
        model_name = f'{base_lm}_{base_model.model}'
    
    # Load testset with initial config, otherwise prompt model to generate it first.
    with open(prompt_path) as f:
        prompt = json.load(f) 
    
    # Load testset with initial config, otherwise prompt model to generate it first.     
    log_dir =  f'logs/game24/{model_name}/'\
                        f'A-star/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    
    # logs storage
    log_dir = log_dir + f'_{model_name}'
    heuristic_fn = None
    # print(os.path.exists(heuristic_log_file))
    if os.path.exists(heuristic_log_file):
        with open(heuristic_log_file) as f:
            heuristic_log = f.read()
        heuristic_fn = load_heuristic_fn(heuristic_log)

    autohd_search(base_model,
               prompt,
               search_algo='beamstar',
               disable_log='False',
               data_path=data_path,
               depth_limit=depth_limit,
               temperature=temperature,
               n_iters=n_iters,
               log_dir=log_dir,
               terminal_beam=num_solutions,
               n_sc = sc_num,
               heuristic_fn=heuristic_fn,
               resume=resume,
               **kwargs)
    
# python -m torch.distributed.run --master_port 12101 --nproc_per_node 1 inference.py --base_lm openai --n_iters 5 --temperature 0.8 --depth_limt 4 
if __name__ == '__main__':
    fire.Fire(main)