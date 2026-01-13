import json
from typing import Literal, Optional

import fire
import utils
from search_config import CalendarPlanToTConfig as CalendarPlanConfig
from world_model import CalenderPlanWorldModel

from reasoners import Reasoner
from reasoners.algorithm import BeamSearch
from reasoners.benchmark import CalendarPlanEvaluator
from reasoners.lm.hf_model import HFModel
from reasoners.lm.openai_model import OpenAIModel


def main(
    base_lm: Literal["hf", "openai", "api"] = "openai",
    model_dir=None,
    api_model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    openai_model="gpt-4o-mini",
    prompt="prompts/calendarplan/prompts.json",
    data_path="data/calendarplan/output_num_{num}.json",
    num_days=5,
    resume=0,
    log_dir=None,
    temperature=0.8,
    beam_size: int = 5,
    calc_reward="llm",
    depth_limit: int = 10,
    hf_peft_path: Optional[str] = None,
    quantized: Optional[Literal["awq", "int8", "fp4", "nf4"]] = None,
    hf_load_awq_path: Optional[str] = None,
    batch_size: int = 1,
    **search_algo_params,
):
    if base_lm == "hf":
        base_model = HFModel(
            model_dir,
            model_dir,
            max_batch_size=batch_size,
            max_new_tokens=512,
            peft_pth=hf_peft_path,
            quantized=quantized,
            load_awq_pth=hf_load_awq_path,
        )
    elif base_lm == "openai":
        base_model = OpenAIModel(openai_model, additional_prompt="NONE")
    elif base_lm == "api":
        from reasoners.lm.llama_api_model import LLaMaApiModel

        base_model = LLaMaApiModel(
            None, None, use_api=True, model_id=api_model_id, quantized=None
        )
        model_dir = base_model.model_id
    else:
        assert False, f"cannot resolve {base_lm=}"

    with open(prompt) as f:
        prompt = json.load(f)

    if base_lm == "hf":
        model_name = model_dir.split("/")[-1]
    else:
        model_name = base_lm

    from datetime import datetime

    log_dir = f"logs/calendarplan/ToT/{datetime.now().strftime('%m%d%Y-%H%M%S')}"
    log_dir = log_dir + f"_{model_name}"

    search_algo_params |= {"max_depth": depth_limit}
    search_algo_params |= {
        "sampling_strategy": "argmax",
        "reward_aggregator": "mean",
        "beam_size": beam_size,
    }
    data_path = data_path.format(num=f"days_{num_days}")

    world_model = CalenderPlanWorldModel(base_model=base_model, prompt=prompt)
    config = CalendarPlanConfig(
        base_model=base_model,
        prompt=prompt,
        temperature=temperature,
        calc_reward=calc_reward,
    )
    search_algo = BeamSearch(**search_algo_params)

    reasoner = Reasoner(
        world_model=world_model, search_config=config, search_algo=search_algo
    )
    evaluator = CalendarPlanEvaluator(
        output_extractor=utils.tot_extractor,
        answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["golden_plan"]),
        init_prompt=prompt,  # will update dynamically
        dataset_path=data_path,
        disable_log=False,
        disable_tqdm=False,
        sample_prompt_type="tot",
        prompt_key="prompt_0shot",
    )

    accuracy = evaluator.evaluate(
        reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir
    )
    print(f"accuracy: {accuracy:.4f}")
    return 0


if __name__ == "__main__":
    fire.Fire(main)
