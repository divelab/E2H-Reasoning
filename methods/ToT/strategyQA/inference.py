import json
import fire
from reasoners.lm import LLaMaApiModel, OpenAIModel, HFModel
from reasoners.lm.llama_api_model import LLaMaApiModel
from typing import Literal
from utils import tot_extractor, retrieve_answer_from_dataset
from datetime import datetime
from reasoners.algorithm import BeamSearch
from world_model import SQAWorldModel
from search_config import SQAConfig
from reasoners import Reasoner
from reasoners.benchmark import StrategyQAEvaluator


def main(base_lm: Literal['hf','openai','api'] = 'hf',
            model_dir: str = None,
            batch_size: int = 1,
            max_seq_len: int = 3072,
            prompt: str = 'prompts/strategyqa/prompts.json',
            data_path: str = 'data/strategyqa/strategyqa_test.json',
            disable_log: bool = False,
            disable_tqdm: bool = False,
            resume: int = 0,
            log_dir: str = None,
            temperature: float = 0.8,
            quantized = 'int8',
            depth_limit: int = 10,
            api_model_id: str = 'meta-llama/Meta-Llama-3.1-70B-Instruct',
            openai_model: str = 'gpt-4o-mini',
            n_beam: int = 5,
            calc_reward: Literal['sampling', 'logits'] = 'sampling',
            **search_algo_params):
    # set base_lm = 'llama' and llama_ckpt = '13B/30B/65B' to use llama with torchscale
    # else set base_lm = 'llama.cpp' and llama_cpp_path = the checkpoint to use llama.cpp

    if base_lm == 'hf':
        base_model = HFModel(model_dir, model_dir,quantized=quantized)
    elif base_lm == 'openai':
        base_model = OpenAIModel(openai_model, additional_prompt="ANSWER")
    elif base_lm == 'api':
        base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None, additional_prompt="ANSWER")
        model_dir = base_model.model_id
    else:
        raise ValueError(f"base_lm {base_lm} is not supported")
        
    with open(prompt) as f:
        prompt = json.load(f)

    log_dir =  f'logs/strategyqa'\
                        f'/tot/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    if base_lm == 'hf' or base_lm == 'api':
        model_name = model_dir.split('/')[-1]
    else:
        model_name = base_lm
    log_dir = log_dir + f'_{model_name}'

    search_algo_params |= {"max_depth": depth_limit, 'beam_size': n_beam}
    search_algo_params |= {
    'sampling_strategy': 'argmax',
    'reward_aggregator': 'mean'
    }

    world_model = SQAWorldModel(base_model=base_model, prompt=prompt)
    config = SQAConfig(base_model=base_model, prompt=prompt, temperature=temperature, depth_limit=depth_limit, calc_reward=calc_reward)
    search_algo = BeamSearch(**search_algo_params)

    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    evaluator = StrategyQAEvaluator(
                 output_extractor= tot_extractor,
                 answer_extractor=lambda x: retrieve_answer_from_dataset(x["answer"]),
                 init_prompt=prompt, # will update dynamically
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="tot",
                 dataset_path=data_path)
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    

if __name__ == '__main__':
    fire.Fire(main)