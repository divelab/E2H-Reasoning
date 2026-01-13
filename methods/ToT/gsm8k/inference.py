import json
from reasoners.lm.openai_model import OpenAIModel
from reasoners.benchmark import GSM8KEvaluator
from reasoners.lm.hf_model import HFModel
import utils
from typing import Literal
import fire
from reasoners.algorithm import BeamSearch
from reasoners import Reasoner
from world_model import GSM8KWorldModel
from search_config import GSM8KConfig
from reasoners.lm.llama_api_model import LLaMaApiModel
from datetime import datetime

def main(base_lm:Literal['hf', 'openai', "api"],
         model_dir="/data3/blakeo/Llama-3.1-8B", 
         prompt="prompts/gsm8k/prompts.json", 
         resume=0, 
         log_dir=None, 
         temperature=0.8, 
         quantized='int8',
         depth_limit: int = 10,
         api_model_id='meta-llama/Meta-Llama-3.1-405B-Instruct',
         openai_model="openai-4o-mini",
         **search_algo_params):

    if base_lm == "openai":
        base_model = OpenAIModel(openai_model, additional_prompt="ANSWER")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir, quantized=quantized)
    elif base_lm == 'api':
        base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None, additional_prompt="ANSWER")
        model_dir = base_model.model_id
    else:
        raise ValueError(f"Unknown base_lm: {base_lm}")
    with open(prompt) as f:
        prompt = json.load(f)

    if base_lm == 'hf':
        model_name= model_dir.split('/')[-1]
    else:
        model_name = base_lm

    log_dir =  f'logs/gsm8k/'\
                        f'ToT/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    log_dir = log_dir + f'_{model_name}'

    search_algo_params |= {"max_depth": depth_limit}
    search_algo_params |= {
    'sampling_strategy': 'argmax',
    'reward_aggregator': 'last'
    }

    world_model = GSM8KWorldModel(base_model=base_model, prompt=prompt)
    config = GSM8KConfig(base_model=base_model, prompt=prompt, temperature=temperature, depth_limit=depth_limit)
    search_algo = BeamSearch(**search_algo_params)

    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    evaluator = GSM8KEvaluator(
                 output_extractor=utils.tot_extractor,
                 answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
                 init_prompt=prompt,
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="tot")

    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    return 0

if __name__ == '__main__':
    fire.Fire(main)
