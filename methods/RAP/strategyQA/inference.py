import pickle
from typing import Type, Callable, Optional

import numpy as np
from reasoners.visualization import TreeLog
from tqdm import tqdm
from datetime import datetime
import json
from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import MCTS, MCTSNode
from reasoners.benchmark import StrategyQAEvaluator
from world_model import StrategyQAWorldModel
from search_config import StrategyQAConfig
import utils
from dataset import get_prompt_examples, get_examples, extract_golden_answer


def node_visualizer(x: MCTSNode):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}

def rap_cum_reward(cum_rewards):
    return sum(cum_rewards) / (len(cum_rewards) + 1)

def rap_strategyQA(base_model: LanguageModel,
              interactive_prompt: dict,
              useful_prompt: dict,
              decompose_prompt: str,
              search_algo: Type[SearchAlgorithm] = MCTS,
              resume: int = 0,
              n_action: int = 4,
              n_confidence: int = 8,
              depth_limit: int = 7,
              force_terminating_on_depth_limit: bool = True,
              batch_size: int = 2,
              temperature: float = 0.8,
              early_stop_base: int = 2,
              early_stop_threshold: float = 0.5,
              reward_alpha: float = 0.5,
              reward_confidence_default: float = 1,
            #   cum_reward: Callable[[list[float]], float] = np.mean,
              cum_reward: Callable[[list[float]], float] = rap_cum_reward,
              calc_q: Callable[[list[float]], float] = max,
              eos_token_id='\n',
              log_dir: Optional[str] = None,
              disable_log: bool = False,
              disable_tqdm: bool = False,
              output_trace_in_each_iter: bool = False,
              **search_algo_params):
    if not disable_log:
        if log_dir is None:
            log_dir = f'logs/strategyQA_{search_algo.__name__}/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        os.makedirs(log_dir, exist_ok=resume >= 0)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)

    search_algo_params |= {'cum_reward': cum_reward, 'calc_q': calc_q, 'disable_tqdm': disable_tqdm, \
                           'output_trace_in_each_iter': output_trace_in_each_iter}
        
    world_model = StrategyQAWorldModel(base_model=base_model, prompt=interactive_prompt,
                                n_confidence=n_confidence, batch_size=batch_size, temperature=temperature, eos_token_id='\n',
                                early_stop_base=early_stop_base, early_stop_threshold=early_stop_threshold)
    config = StrategyQAConfig(base_model=base_model, prompt=interactive_prompt, useful_prompt=useful_prompt, decompose_prompt=decompose_prompt,
                         n_actions=n_action, batch_size=batch_size, temperature=temperature, eos_token_id='\n',
                         reward_alpha=reward_alpha, reward_confidence_default=reward_confidence_default,
                         force_terminating_on_depth_limit=force_terminating_on_depth_limit, depth_limit=depth_limit)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    evaluator = StrategyQAEvaluator(
                 output_extractor= lambda x: utils.extract_final_answer(x.terminal_state[-1]),
                 answer_extractor=lambda x: x["answer"],
                 init_prompt=interactive_prompt, # will update dynamically
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="rap",
                 dataset_path='data/strategyqa/strategyqa_test.json')
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, resume=resume, resume=resume, log_dir=log_dir)
    print(f'Acc: {accuracy}')


if __name__ == '__main__':
    import os
    import sys
    import json
    import fire
    from reasoners.lm import HFModel

    def main(base_lm: str = 'hf', #llama means llama_v1 and llama2 means llama_v2
             model_dir: str = None,
             quantized: str = None,
             batch_size: int = 2,
             max_seq_len: int = 2048,
             interactive_prompt: str = 'methods/RAP/strategyQA/prompts/interactive_examples-1.json',
             useful_prompt: str = 'methods/RAP/strategyQA/prompts/useful_examples-1.json',
             decompose_prompt: str = 'methods/RAP/strategyQA/prompts/problem_decompose_examples-1.0.1.txt',
             disable_log: bool = False,
             disable_tqdm: bool = False,
             **kwargs):
        # set base_lm = 'llama' and llama_ckpt = '13B/30B/65B' to use llama with torchscale
        # else set base_lm = 'llama.cpp' and llama_cpp_path = the checkpoint to use llama.cpp

        with open(interactive_prompt) as f:
            interactive_prompt = json.load(f)
        with open(useful_prompt) as f:
            useful_prompt = json.load(f)
        decompose_prompt = get_prompt_examples(path=decompose_prompt)
        
        if base_lm == "hf":
            base_model = HFModel(model_pth=model_dir, tokenizer_pth=model_dir, quantized=quantized, max_batch_size=batch_size)
        else:
            assert False, f'cannot resolve {base_lm=}'
            
        if base_lm == 'hf':
            model_name= model_dir.split('/')[-1]
        else:
            model_name = base_model
        print(model_name)
        log_dir =  f'logs/strategyQA/'\
                        f'/RAP/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}_{model_name}'
        rap_strategyQA(base_model=base_model,
                  interactive_prompt=interactive_prompt,
                  useful_prompt=useful_prompt,
                  decompose_prompt=decompose_prompt,
                  batch_size=batch_size,
                  disable_log=disable_log,
                  disable_tqdm=disable_tqdm,
                  log_dir = log_dir,
                  **kwargs)
        
    fire.Fire(main)
