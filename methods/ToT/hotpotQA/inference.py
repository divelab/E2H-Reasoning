import json
from typing import Literal

import fire
import utils
from search_config import HotpotConfig
from world_model import HotpotWorldModel

from reasoners import Reasoner
from reasoners.algorithm import BeamSearch
from reasoners.benchmark import HotpotQAEvaluator
from reasoners.lm.hf_model import HFModel
from reasoners.lm.llama_api_model import LLaMaApiModel
from reasoners.lm.openai_model import OpenAIModel


def main(
    base_lm: Literal["hf", "openai", "api"],
    model_dir=None,
    api_model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    openai_model="gpt-4o-mini",
    prompt="prompts/hotpotQA/prompts.json",
    resume=0,
    log_dir=None,
    temperature=0.8,
    calc_reward="llm",
    beam_size: int = 5,
    quantized="int8",
    depth_limit: int = 10,
    **search_algo_params,
):
    if base_lm == "openai":
        base_model = OpenAIModel(openai_model)
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir, quantized=quantized)
    elif base_lm == "api":
        base_model = LLaMaApiModel(
            None, None, use_api=True, model_id=api_model_id, quantized=None
        )
        model_dir = base_model.model_id
    else:
        raise ValueError(f"Unknown base_lm: {base_lm}")
    with open(prompt) as f:
        prompt = json.load(f)

    if base_lm == "hf":
        model_name = model_dir.split("/")[-1]
    else:
        model_name = base_lm

    from datetime import datetime

    log_dir = f"logs/hotpotQA/ToT/{datetime.now().strftime('%m%d%Y-%H%M%S')}"
    log_dir = log_dir + f"_{model_name}"

    search_algo_params |= {"max_depth": depth_limit}
    search_algo_params |= {
        "sampling_strategy": "argmax",
        "reward_aggregator": "mean",
        "beam_size": beam_size,
    }

    world_model = HotpotWorldModel(base_model=base_model, prompt=prompt)
    config = HotpotConfig(
        base_model=base_model,
        prompt=prompt,
        temperature=temperature,
        calc_reward=calc_reward,
        depth_limit=depth_limit,
    )
    search_algo = BeamSearch(**search_algo_params)

    reasoner = Reasoner(
        world_model=world_model, search_config=config, search_algo=search_algo
    )
    evaluator = HotpotQAEvaluator(
        output_extractor=utils.tot_extractor,
        answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
        init_prompt=prompt,  # will update dynamically
        disable_log=False,
        disable_tqdm=False,
        sample_prompt_type="cot",
        num_shot=4,
    )

    accuracy = evaluator.evaluate(
        reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir
    )
    print(f"accuracy: {accuracy:.4f}")
    return 0


if __name__ == "__main__":
    fire.Fire(main)
