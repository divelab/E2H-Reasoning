from typing import Literal
import json
from reasoners.benchmark import AQuAEvaluator
import utils
import fire
from reasoners.lm.hf_model import HFModel
from reasoners.lm.openai_model import OpenAIModel
from reasoners.algorithm import BeamSearch
from world_model import AQUAWorldModel
from search_config import AQUAConfig
from reasoners import Reasoner
from datetime import datetime


def main(base_lm:Literal['hf', 'openai'],
         model_dir= None, 
         prompt="prompts/aqua/prompts.json", 
         data_path="data/aqua", 
         datasetname="test",
         quantized='int8',
         resume=0, 
         temperature=0.8,
         log_dir=None,
         depth_limit: int = 10,
         calc_rewards: Literal["sampling", "logits"] = "sampling",
         openai_model = "gpt-4o-mini",
         **search_algo_params):

    if base_lm == "openai":
        base_model = OpenAIModel(openai_model, additional_prompt="ANSWER")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir, quantized=quantized)
    else:
        raise ValueError(f"base_lm {base_lm} is not supported")
    with open(prompt) as f:
        prompt = json.load(f)

    search_algo_params |= {"max_depth": depth_limit}
    search_algo_params |= {
    'sampling_strategy': 'argmax',
    'reward_aggregator': 'mean'
    }


    log_dir =  f'logs/AQuA/'\
                        f'tot/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    if base_lm == 'hf':
        model_name= model_dir.split('/')[-1]
    else:
        model_name = base_lm
    log_dir = log_dir + f'_{model_name}'

    world_model = AQUAWorldModel(base_model=base_model, prompt=prompt)
    config = AQUAConfig(base_model=base_model, prompt=prompt, temperature=temperature, depth_limit=depth_limit, calc_reward=calc_rewards)
    search_algo = BeamSearch(**search_algo_params)

    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    evaluator = AQuAEvaluator(
                 output_extractor=utils.tot_extractor,
                 answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
                 init_prompt=prompt,
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="tot",
                 dataset_path=data_path,
                 datasetname=datasetname)
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    return 0



if __name__ == '__main__':
    fire.Fire(main)