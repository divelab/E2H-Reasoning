import json
import fire
from typing import Any
import json
from typing import Optional, Literal

from data.prontoqa.dataset import ProntoQADataset
from reasoners import Reasoner
from reasoners.algorithm import BeamSearch, DFS
from reasoners.benchmark import ProntoQAEvaluatorFinal
from reasoners.lm import OpenAIModel
from reasoners.lm import LLaMaApiModel
from world_model import ProntoQAToTWorldModel
from search_config import ProntoQAToTSearchConfig
from utils import dfs_bw_extractor, bfs_pronto_extractor
ProntoQAState = list[str]
ProntoQAAction = str
    

def main(
           model_dir: str = None,
           base_lm: Literal['api', 'hf', 'openai']  = 'openai',
           batch_size = 4,
           search_algo: str = "beam",
           resume: int = 0,
           depth_limit: int = 6,
           log_dir: Optional[str] = None,
           temperature: float = 0.8,
           api_model_id='meta-llama/Meta-Llama-3.1-405B-Instruct',
           openai_model="gpt-4o-mini",
           **search_algo_params):

    if search_algo == "beam":
        search_algo_params |= {"max_depth": depth_limit}
    elif search_algo == "dfs":
        search_algo_params |= {"depth": depth_limit}
    else:
        print("Unknown search algorithm", search_algo)
        raise NotImplementedError

    if base_lm == 'openai':
        base_model = OpenAIModel(openai_model, additional_prompt="CONTINUE")
    elif base_lm == 'hf':
        from reasoners.lm import HFModel
        base_model = HFModel(model_dir, model_dir, max_batch_size=batch_size, max_new_tokens=512)
    elif base_lm == 'api':
        base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None, additional_prompt="CONTINUE")
        model_dir = base_model.model_id
    else:
        raise ValueError(f"Unknown base_lm: {base_lm}")


    world_model = ProntoQAToTWorldModel()
    search_config = ProntoQAToTSearchConfig(base_model=base_model, temperature=temperature, calc_reward="sampling")    
    output_extractor = dfs_bw_extractor if search_algo == "dfs" else bfs_pronto_extractor
    if search_algo == "dfs":
        search_algo = DFS(**search_algo_params)
    elif search_algo == "beam":
        search_algo = BeamSearch(**search_algo_params)
    else:
        raise NotImplementedError
   
    with open('prompts/prontoqa/example_next_steps.json') as f:
        init_prompt = json.load(f)
    
    reasoner = Reasoner(world_model=world_model, search_config=search_config, search_algo=search_algo)
    evaluator = ProntoQAEvaluatorFinal(
        init_prompt=init_prompt['next_steps'],
        sample_prompt_type="cot",
        disable_log=False,
        disable_tqdm=False, dataset = ProntoQADataset.from_file(
            'data/prontoqa/345hop_random_true.json'
        ),
        output_extractor=output_extractor,
        answer_extractor=lambda x: "\n".join(x.test_example.chain_of_thought[2::2])
    )

    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(accuracy)

if __name__ == '__main__':
    fire.Fire(main)