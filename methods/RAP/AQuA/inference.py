import pickle
from typing import Type, Callable, Optional , Literal
import fire
import os
import numpy as np
from reasoners.algorithm.mcts import MCTSResult
from regex import F
#from sklearn import base
from tqdm import tqdm
from datetime import datetime

from reasoners.lm import HFModel
from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import MCTS, MCTSNode, MCTSAggregation
from reasoners.visualization import TreeLog
from reasoners.benchmark import AQuAEvaluator

from world_model import MATHWorldModel, MATHState, MATHAction
from search_config import MATHConfig
import utils

def eval_non_aggregate(pkl_pth:str, resume_s:int, resume_e:int):
        evaluator = AQuAEvaluator(output_extractor=utils.retrieve_answer,
                               answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
                               init_prompt=None,
                               sample_prompt_type="rap",
                               disable_log=None,
                               disable_tqdm=None)
        data = list(evaluator.full_dataset)[resume_s:resume_e]
        correct_count = 0
        for i in range(resume_s, resume_e):
            case_result_pure = pickle.load(open(os.path.join(pkl_pth, f'{i+1}.pkl'), 'rb'))
            case_result_pure = MCTSResult(
                terminal_state=case_result_pure.terminal_state,
                cum_reward=case_result_pure.cum_reward,
                trace=case_result_pure.trace,
                trace_of_nodes=case_result_pure.trace_of_nodes,
                tree_state=case_result_pure.tree_state,
                trace_in_each_iter=case_result_pure.trace_in_each_iter,
                tree_state_after_each_iter=case_result_pure.tree_state_after_each_iter,
                aggregated_result=None,
            )

            output = evaluator.output_extractor(case_result_pure)
            answer = evaluator.answer_extractor(data[i])
            correct = evaluator.eval_output(answer, output)
            correct_count += correct
            accuracy = correct_count / (i + 1)
            log_str = f'Case #{resume_s + i + 1}: {correct=}, {output=}, {answer=};'\
                        f'{accuracy=:.3f} ({correct_count}/{i + 1})'
            with open(os.path.join(pkl_pth, 'non_aggr_result.log'), 'a') as f:
                print(log_str, file=f)
                
def eval_aggregate(pkl_pth:str, resume_s:int, resume_e:int):
    evaluator = AQuAEvaluator(output_extractor=utils.retrieve_answer,
                        answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
                        init_prompt=None,
                        sample_prompt_type="rap",
                        disable_log=None,
                        disable_tqdm=None)
    data = list(evaluator.full_dataset)[resume_s:resume_e]
    correct_count = 0
    for i in range(resume_s, resume_e):
        aggregator = MCTSAggregation(evaluator.output_extractor, weight_policy='edge')
        case_result_pure = pickle.load(open(os.path.join(pkl_pth, f'{i+1}.pkl'), 'rb'))
        output = aggregator(case_result_pure.tree_state)
        answer = evaluator.answer_extractor(data[i])
        correct = evaluator.eval_output(answer, output)
        correct_count += correct
        accuracy = correct_count / (i + 1)
        log_str = f'Case #{resume_s + i + 1}: {correct=}, {output=}, {answer=};'\
                    f'{accuracy=:.3f} ({correct_count}/{i + 1})'
        with open(os.path.join(pkl_pth, 'aggr_result.log'), 'a') as f:
            print(log_str, file=f)




def node_visualizer(x: MCTSNode):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}


def rap_AQuA(base_model: LanguageModel,
              prompt: dict,
              useful_prompt: dict,
              search_algo: Type[SearchAlgorithm] = MCTS,
              resume: int = 0,
              n_action: int = 4,
              n_confidence: int = 8,
              depth_limit: int = 10,
              force_terminating_on_depth_limit: bool = True,
              batch_size: int = 1,
              temperature: float = 0.8,
              early_stop_base: int = 2,
              early_stop_threshold: float = 0.5,
              reward_alpha: float = 0.5,
              reward_confidence_default: float = 0.8,
              cum_reward: Callable[[list[float]], float] = np.mean,
              calc_q: Callable[[list[float]], float] = max,
              log_dir: Optional[str] = None,
              disable_log: bool = False,
              disable_tqdm: bool = False,
              output_trace_in_each_iter: bool = True,
              aggregate: bool = True,
              weight_policy:str = 'edge',
              data_path="data/aqua/", 
              datasetname="test",
              **search_algo_params):
    
    print(f'aggregate: {aggregate}, weight_policy: {weight_policy}')
    if aggregate:
        aggregator = MCTSAggregation(utils.retrieve_answer, weight_policy=weight_policy)
    else:
        aggregator = None
    
    search_algo_params |= {'cum_reward': cum_reward, 
                           'calc_q': calc_q, 
                           'disable_tqdm': disable_tqdm, 
                           'output_trace_in_each_iter': output_trace_in_each_iter,
                           'node_visualizer': node_visualizer, 
                           'aggregator': aggregator,
                           'w_exp': 1.0,
                           'depth_limit': depth_limit 
                           }
    
    world_model = MATHWorldModel(
        base_model=base_model,
        n_confidence=n_confidence, 
        batch_size=batch_size, 
        temperature=temperature,
        early_stop_base=early_stop_base, 
        early_stop_threshold=early_stop_threshold,
        score_prompts="examples/RAP/AQuA/prompts/score_examples.json")
    
    config = MATHConfig(
        base_model=base_model, 
        useful_prompt=useful_prompt,
        n_actions=n_action, 
        batch_size=batch_size, 
        temperature=temperature,
        reward_alpha=reward_alpha, 
        reward_confidence_default=reward_confidence_default,
        force_terminating_on_depth_limit=force_terminating_on_depth_limit, 
        depth_limit=depth_limit)
    
    search_algo = search_algo(**search_algo_params)
    
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    evaluator = AQuAEvaluator(output_extractor=utils.retrieve_answer,
                               answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
                               init_prompt=prompt,
                               sample_prompt_type="rap",
                               disable_log=disable_log,
                               disable_tqdm=disable_tqdm,
                               dataset_path=data_path,
                               datasetname=datasetname)
    accuracy = evaluator.evaluate(reasoner, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    return 0

if __name__ == '__main__':
    import json
    import fire

    def main(
        model_dir = '/path/to/model',
        llama_size = None,
        lora_dir = None,
        batch_size = 1,
        mem_map = [16,22],
        prompt = "examples/RAP/AQuA/prompts/prompt_pool.json",
        useful_prompt: str = 'examples/RAP/AQuA/prompts/useful_examples.json',
        disable_log = False,
        disable_tqdm = False,
        reward_alpha = 0.5,
        weight_policy:str = 'edge',
        resume:int = 0,
        data_path="data/aqua/", 
        datasetname="test",
        quantized=None,
        **kwargs):

        base_model = HFModel(model_pth=model_dir, tokenizer_pth=model_dir, quantized=quantized, max_batch_size=batch_size)
        
        with open(prompt) as f:
            prompt = json.load(f)
        with open(useful_prompt) as f:
            useful_prompt = json.load(f)
        rap_AQuA(
             base_model=base_model,
             prompt=prompt,
             useful_prompt=useful_prompt,
             batch_size=batch_size,
             disable_log=disable_log,
             disable_tqdm=disable_tqdm,
             reward_alpha = reward_alpha,
             weight_policy=weight_policy,
             resume=resume,
             data_path=data_path, 
             datasetname=datasetname,
             **kwargs)

    fire.Fire(main)
