from datetime import datetime

import json
import fire
import os
from typing import Literal
import json
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.hf_model import HFModel
from world_model import TripPlanWorldModel
from search_config import TPConfig
from reasoners.algorithm import BeamSearch
from reasoners import Reasoner
import utils
from reasoners.benchmark import TripPlanEvaluator
from reasoners.lm.llama_api_model import LLaMaApiModel


def main(base_lm:Literal['hf', 'openai', 'api',] = 'api',
         model_dir= "/data3/blakeo/Llama-3.1-8B", 
         num_cities=3, 
         data_path="data/tripplan/test_TripPlan-cities-{num_cities}.json",
         prompt_path="prompts/tripplan/prompts.json", 
         quantized='int8',
         resume=0, 
         temperature=0.8,
         log_dir=None,
         depth_limit: int = 6,
         openai_model="gpt-4o-mini",
         **search_algo_params):

    if base_lm == "openai":
        base_model = OpenAIModel(openai_model, additional_prompt="NONE")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir,quantized=quantized)
    elif base_lm == 'api':
        base_model = LLaMaApiModel(None, None, use_api=True, model_id='meta-llama/Meta-Llama-3.1-405B-Instruct', quantized=None)
        model_dir = base_model.model_id
    else:
        raise ValueError(f"base_lm {base_lm} is not supported")
    
    if base_lm == 'hf':
        model_name = model_dir.split('/')[-1]
    else:
        model_name = f'{base_lm}_{base_model.model}'
    data_path = data_path.format(num_cities=num_cities)
    prompt_path = prompt_path.format(num_cities=num_cities)
    with open(prompt_path) as f:
        prompt = json.load(f) 
        
    log_dir =  f'logs/tripplan/num_cities-{num_cities}/'\
                        f'ToT/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    
    search_algo_params |= {"max_depth": depth_limit}
    search_algo_params |= {
    'sampling_strategy': 'argmax',
    'reward_aggregator': 'mean'
    }

    # logs storage
    log_dir = log_dir + f'_{model_name}'
    world_model = TripPlanWorldModel(total_days=100, base_model=base_model, prompt=prompt)
    config = TPConfig(base_model=base_model, prompt=prompt, temperature=temperature, calc_reward="sampling")
    search_algo = BeamSearch(**search_algo_params)

    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    evaluator = TripPlanEvaluator(
        output_extractor=utils.tot_extractor,
        answer_extractor=utils.retrieve_answer_from_dataset,
        init_prompt=prompt,
        disable_log=False,
        disable_tqdm=False,
        sample_prompt_type="tot",
        dataset_path=data_path,
        num_cities=num_cities
    )
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir, tripPlan=True)
    print(f'accuracy: {accuracy:.4f}')
    

if __name__ == '__main__':
    fire.Fire(main)