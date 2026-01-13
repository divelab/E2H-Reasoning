from typing import Type, Callable, Optional
from reasoners.lm import HFModel
import re
import numpy as np

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.benchmark import BWEvaluator
from reasoners.algorithm import MCTS

from world_model import BlocksWorldModel
from search_config import BWConfig


def extract_step(file_path):
    match = re.search(r"_step_(\d+)_", file_path)
    return int(match.group(1)) if match else None


def RAP_bw(
    base_model: LanguageModel,
    prompt: dict,
    search_algo: Type[SearchAlgorithm] = MCTS,
    data_path: str = "data",
    resume: int = 0,
    depth_limit: int = 6,
    reward_alpha: float = 0.5,
    batch_size=1,
    goal_reached_reward=100,
    goal_reward_default=0.0,
    cum_reward: Callable[[list[float]], float] = sum,
    calc_q: Callable[[list[float]], float] = np.mean,
    log_dir: Optional[str] = None,
    disable_log: bool = False,
    domain_file: str = "",
    config_file: str = "",
    lm_plan_file: str = "lm_plan.tmp",
    **search_algo_params,
):
    search_algo_params |= {
        "cum_reward": cum_reward,
        "calc_q": calc_q,
        "depth_limit": depth_limit,
        "disable_tqdm": False,
    }
    world_model = BlocksWorldModel(
        base_model=base_model,
        prompt=prompt,
        batch_size=batch_size,
        max_steps=depth_limit,
    )
    config = BWConfig(
        base_model=base_model,
        prompt=prompt,
        batch_size=batch_size,
        reward_alpha=reward_alpha,
        goal_reached_reward=goal_reached_reward,
        goal_reward_default=goal_reward_default,
    )
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(
        world_model=world_model, search_config=config, search_algo=search_algo
    )
    evaluator = BWEvaluator(
        config_file=config_file,
        domain_file=domain_file,
        data_path=data_path,
        init_prompt=prompt,
        disable_log=disable_log,
    )
    accuracy = evaluator.evaluate(
        reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir
    )
    print(accuracy)


if __name__ == "__main__":
    import json
    import fire
    import random
    import torch
    import torch.backends.cudnn

    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    def main(
        model_dir="/path/to/model",
        peft_path=None,
        disable_log: bool = False,
        config_file: str = "methods/CoT/blocksworld/data/bw_config.yaml",
        domain_file: str = "methods/CoT/blocksworld/data/generated_domain.pddl",
        lm_plan_file: str = "lm_plan.tmp",
        quantized="None",  # awq, int8, fp4, nf4, None
        load_awq_pth=None,
        steps=2,
        **kwargs,
    ):
        # maybe other transformer models also support, we have not check that
        data_path = f"data/split_v2/split_v2_step_{steps}_data.json"
        prompt_path = f"prompts/blocksworld/pool_prompt_v2_step_{steps}.json"
        depth_limit = steps
        with open(prompt_path) as f:
            prompt = json.load(f)
        device = torch.device("cuda:0")
        llama_model = HFModel(
            model_dir,
            model_dir,
            device=device,
            max_batch_size=1,
            max_new_tokens=512,
            quantized=quantized,
            peft_pth=peft_path,
            load_awq_pth=load_awq_pth,
        )

        RAP_bw(
            llama_model,
            prompt,
            disable_log=disable_log,
            data_path=data_path,
            config_file=config_file,
            domain_file=domain_file,
            depth_limit=depth_limit,
            lm_plan_file=lm_plan_file,
            **kwargs,
        )


    fire.Fire(main)  # user will need to switch the model in the code
