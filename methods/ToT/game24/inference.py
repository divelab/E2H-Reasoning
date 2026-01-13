from typing import Type, Optional, Literal
import json
import fire
from datetime import datetime
from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import BeamSearch, MCTS, MCTSNode
from reasoners.visualization import TreeLog
from reasoners.benchmark import Game24Evaluator

from world_model import Game24WorldModel
from search_config import Game24Config
import utils


def tot_game24(base_model: LanguageModel,
               prompts: dict,
               search_algo: Type[SearchAlgorithm] = BeamSearch,
               resume: int = 0,
               n_action: int = 4,
               n_beam: int = 5,
               n_eval: int = 3,
               depth_limit: int = 4,
               batch_size: int = 3,
               log_dir: Optional[str] = None,
               disable_log: bool = False,
               calc_reward: Literal['sampling', 'logits'] = 'sampling',
               **search_algo_params):
    ## keep the best 5 candidates, need at most 4 steps to solve
    ## following ToT, eval step will consider number of times to prompt for state evaluation
    # search_algo_params |= {'beam_size': n_beam, 'max_depth': depth_limit}
    if search_algo.__name__ == 'BeamSearch':
        search_algo_params['beam_size'] = n_beam
        search_algo_params['max_depth'] = depth_limit
    search_algo_params |= {'output_trace_in_each_iter': True, 'depth_limit': depth_limit, 'disable_tqdm': False}
    world_model = Game24WorldModel(base_model=base_model, prompt=prompts, batch_size=batch_size)
    config = Game24Config(base_model=base_model, prompt=prompts, calc_reward=calc_reward,
                          n_actions=n_action, n_eval=n_eval, batch_size=batch_size, depth_limit=depth_limit,)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    evaluator = Game24Evaluator(
                                  prompt=prompts, 
                                  disable_log=disable_log, 
                                  output_extractor=utils.parse_output,
                                  answer_extractor=lambda x: (24.0, x),
                                  input_processor=lambda x: x,
                                  sample_prompt_type="tot",
                                  heuristic_search=True)
    metric = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=1, resume=resume, log_dir=log_dir)
    print(f'Acc: {metric}')



def main(base_lm: Literal['hf', 'openai', 'api'] = 'openai',
             model_dir: str = '',
             hf_peft_path: Optional[str] = None,
             quantized: Optional[Literal['awq', 'int8', 'fp4', 'nf4']] = None,
             hf_load_awq_path: Optional[str] = None,
             batch_size: int = 1,
             prompts: str = 'prompts/game24/prompts.json',
             disable_log: bool = False,
             disable_tqdm: bool = False,
             api_model_id='meta-llama/Meta-Llama-3.1-70B-Instruct',
             openai_model='gpt-4o-mini',
             **kwargs):
        with open(prompts) as f:
            prompts = json.load(f)

        if base_lm == 'hf':
            from reasoners.lm import HFModel
            base_model = HFModel(model_dir, model_dir, max_batch_size=batch_size, max_new_tokens=512,
                                 peft_pth=hf_peft_path, quantized=quantized, load_awq_pth=hf_load_awq_path)
        elif base_lm == 'openai':
            from reasoners.lm.openai_model import OpenAIModel
            base_model = OpenAIModel(openai_model, additional_prompt="NONE")
        elif base_lm == 'api':
            from reasoners.lm.llama_api_model import LLaMaApiModel
            base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None)
            model_dir = base_model.model_id
        else:
            assert False, f'cannot resolve {base_lm=}'
        
        log_dir =  f'logs/game24'\
                        f'/tot/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        if base_lm == 'hf' or base_lm == 'api':
            model_name = model_dir.split('/')[-1]
        else:
            model_name = base_lm
        log_dir = log_dir + f'_{model_name}'
        
        
        tot_game24(base_model=base_model,
                   prompts=prompts,
                   batch_size=batch_size,
                   n_beam=5,
                   disable_log=disable_log,
                   search_algo=BeamSearch,
                   log_dir=log_dir,
                   **kwargs)




if __name__ == '__main__':
    fire.Fire(main)
