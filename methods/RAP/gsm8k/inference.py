from typing import Type, Callable, Optional, Literal

import numpy as np

from reasoners.benchmark import GSM8KEvaluator

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import MCTS, MCTSNode, MCTSAggregation

from world_model import GSM8kWorldModel, GSM8kPromptDict
from search_config import GSM8kConfig, GSM8kUsefulPrompt
import utils
from reasoners.lm.llama_api_model import LLaMaApiModel
from datetime import datetime


def node_visualizer(x: MCTSNode):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}


def rap_gsm8k(
    base_model: LanguageModel,
    prompt: GSM8kPromptDict,
    useful_prompt: GSM8kUsefulPrompt,
    search_algo: Type[SearchAlgorithm] = MCTS,
    resume: int = 0,
    n_action: int = 4,
    n_confidence: int = 8,
    depth_limit: int = 5,
    force_terminating_on_depth_limit: bool = True,
    batch_size: int = 2,
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
    **search_algo_params,
):
    if aggregate:
        aggregator = MCTSAggregation(utils.retrieve_answer, weight_policy="edge")
    else:
        aggregator = None

    search_algo_params |= {
        "cum_reward": cum_reward,
        "calc_q": calc_q,
        "disable_tqdm": disable_tqdm,
        "output_trace_in_each_iter": output_trace_in_each_iter,
        "node_visualizer": node_visualizer,
        "aggregator": aggregator,
        "depth_limit": depth_limit,
    }
    world_model = GSM8kWorldModel(
        base_model=base_model,
        n_confidence=n_confidence,
        batch_size=batch_size,
        temperature=temperature,
        early_stop_base=early_stop_base,
        early_stop_threshold=early_stop_threshold,
    )
    config = GSM8kConfig(
        base_model=base_model,
        useful_prompt=useful_prompt,
        n_actions=n_action,
        batch_size=batch_size,
        temperature=temperature,
        reward_alpha=reward_alpha,
        reward_confidence_default=reward_confidence_default,
        force_terminating_on_depth_limit=force_terminating_on_depth_limit,
        depth_limit=depth_limit,
    )
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(
        world_model=world_model, search_config=config, search_algo=search_algo
    )

    evaluator = GSM8KEvaluator(
        output_extractor=utils.retrieve_answer,
        answer_extractor=utils.retrieve_answer_from_dataset,
        init_prompt=prompt,
        sample_prompt_type="rap",
        disable_log=disable_log,
        disable_tqdm=disable_tqdm,
    )
    accuracy = evaluator.evaluate(reasoner, num_shot=4, resume=resume, log_dir=log_dir)
    print(accuracy)


if __name__ == "__main__":
    import os
    import sys
    import json
    import warnings
    import fire
    import random

    llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
    llama_2_ckpts = os.environ.get("LLAMA2_CKPTS", None)
    llama_3_ckpts = os.environ.get("LLAMA3_CKPTS", None)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        sys.stdout = open(os.devnull, "w")
        warnings.filterwarnings("ignore")

    def main(
        base_lm: Literal["hf"] = "hf",
        model_dir: str = "/data3/blakeo/Llama-3.1-8B",
        quantized: Optional[Literal["awq", "int8", "fp4", "nf4", "None"]] = None,
        hf_load_awq_path: Optional[str] = None,
        batch_size: int = 1,
        useful_prompt: str = "methods/RAP/gsm8k/prompts/useful_examples.json",
        prompt: str = "methods/RAP/gsm8k/prompts/prompt_pool.json",
        disable_log: bool = False,
        disable_tqdm: bool = False,
        #  api_model_id='meta-llama/Meta-Llama-3.1-8B-Instruct',
        **kwargs,
    ):
        with open(useful_prompt) as f:
            useful_prompt = json.load(f)
        with open(prompt) as f:
            prompt = json.load(f)
        if base_lm in ["llama", "llama-2", "llama-3"]:
            import torch
            import torch.backends.cudnn

            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True

        if base_lm == "hf":
            from reasoners.lm import HFModel

            # model_pth=model_dir, tokenizer_pth=model_dir, quantized=quantized, max_batch_size=batch_size
            base_model = HFModel(
                model_pth=model_dir,
                tokenizer_pth=model_dir,
                quantized=quantized,
                max_batch_size=batch_size,
            )
        else:
            assert False, f"cannot resolve {base_lm=}"

        log_dir = f"logs/gsm8k//RAP/{datetime.now().strftime('%m%d%Y-%H%M%S')}"
        if base_lm == "hf":
            model_name = model_dir.split("/")[-1]
        else:
            model_name = base_lm
        log_dir = log_dir + f"_{model_name}"
        print(log_dir)
        rap_gsm8k(
            base_model=base_model,
            useful_prompt=useful_prompt,
            prompt=prompt,
            batch_size=batch_size,
            disable_log=disable_log or local_rank != 0,
            disable_tqdm=disable_tqdm or local_rank != 0,
            log_dir=log_dir,
            **kwargs,
        )
        # tokens_log_dir =  f'logs/gsm8k/'\
        #                     f'RAP/8B/'\
        #                     f'TokensGenerated'

        # os.makedirs(tokens_log_dir, exist_ok=True)
        # log_file_path = os.path.join(tokens_log_dir, "tokens_generated.txt")
        # with open(log_file_path, "w") as log_file:
        #     log_file.write(f"Tokens generated: {base_model.tokens_generated}\n")

    fire.Fire(main)
