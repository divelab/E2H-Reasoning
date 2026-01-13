from datetime import datetime

import json
import fire
import os
from typing import Sequence, Any, Literal
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.hf_model import HFModel
from reasoners.lm.gemini_model import BardCompletionModel
from reasoners.lm.anthropic_model import ClaudeModel
from reasoners.lm.llama_api_model import LLaMaApiModel
from reasoners.benchmark import TripPlanEvaluator
import utils


class IOReasoner:
    def __init__(self, base_model, temperature=0.8):
        self.base_model = base_model
        self.temperature = temperature

    def __call__(self, example, prompt=None):
        inputs = prompt["o1"].replace("{Question}", example)
        # print("\nModel input: ", inputs)

        do_sample = True
        if (
            isinstance(self.base_model, OpenAIModel)
            or isinstance(self.base_model, BardCompletionModel)
            or isinstance(self.base_model, ClaudeModel)
        ):
            eos_token_id = []

        output = (
            self.base_model.generate(
                [inputs],
                hide_input=True,
                do_sample=do_sample,
                temperature=self.temperature,
                eos_token_id=eos_token_id,
            )
            .text[0]
            .strip()
        )
        return output


def main(
    base_lm: Literal["hf", "openai", "api"],
    model_dir=None,
    num_cities=3,
    data_path="data/tripplan/test_TripPlan-cities-{num_cities}.json",
    prompt_path="examples/IO/tripplan/prompt.json",
    quantized="int8",
    resume=0,
    temperature=0,
    api_model_id="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    log_dir=None,
):
    if base_lm == "openai":
        base_model = OpenAIModel("o1", additional_prompt="NONE")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir, quantized=quantized)
    elif base_lm == "api":
        base_model = LLaMaApiModel(
            None, None, use_api=True, model_id=api_model_id, quantized=None
        )
        model_dir = base_model.model_id
    else:
        raise ValueError(f"base_lm {base_lm} is not supported")

    if base_lm == "hf" or base_lm == "api":
        model_name = model_dir.split("/")[-1]
    else:
        model_name = f"{base_lm}_{base_model.model}"

    data_path = data_path.format(num_cities=num_cities)
    prompt_path = prompt_path.format(num_cities=num_cities)
    # Load testset with initial config, otherwise prompt model to generate it first.

    with open(prompt_path) as f:
        prompt = json.load(f)

    log_dir = (
        f"logs/tripplan/num_cities-{num_cities}/"
        f"o1/"
        f"{datetime.now().strftime('%m%d%Y-%H%M%S')}"
    )

    # logs storage
    log_dir = log_dir + f"_{model_name}"

    reasoner = IOReasoner(base_model, temperature=temperature)
    evaluator = TripPlanEvaluator(
        output_extractor=utils.parse_response,
        answer_extractor=utils.retrieve_answer_from_dataset,
        init_prompt=prompt,  # will update dynamically
        disable_log=False,
        disable_tqdm=False,
        sample_prompt_type="o1",
        dataset_path=data_path,
    )
    accuracy = evaluator.evaluate(
        reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir
    )
    print(f"accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
