from datetime import datetime

import json
import fire
from typing import Sequence, Any, Literal
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.hf_model import HFModel
from reasoners.benchmark import Game24Evaluator
from reasoners.lm import LLaMaApiModel
import utils


class IOReasoner:
    def __init__(self, base_model, temperature=0.8):
        self.base_model = base_model
        self.temperature = temperature

    def __call__(self, example, prompt=None):
        inputs = prompt["o1"].replace("{input}", example)
        print("inputs: ", inputs)

        do_sample = True
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


# num_days will always come first
def main(
    base_lm: Literal["hf", "openai", "api"],
    model_dir=None,
    quantized="int8",
    resume=0,
    temperature=0,
    log_dir=None,
):
    
    if base_lm == "openai":
        base_model = OpenAIModel("o1-mini", additional_prompt="NONE")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir, quantized=quantized)
    elif base_lm == 'api':
        base_model = LLaMaApiModel(None, None, use_api=True, model_id="meta-llama/Meta-Llama-3.1-70B-Instruct", quantized=None)
        model_dir = base_model.model_id
    else:
        raise ValueError(f"base_lm {base_lm} is not supported")

    if base_lm == "hf":
        model_name = model_dir.split("/")[-1]
    else:
        model_name = f"{base_lm}_{base_model.model}"

    log_dir = f"logs/game24/o1-mini/4o/{datetime.now().strftime('%m%d%Y-%H%M%S')}"
    log_dir = log_dir + f"_{model_name}"

    with open('prompts/game24/prompts.json') as f:
        prompt = json.load(f)

    reasoner = IOReasoner(base_model, temperature=temperature)
    evaluator = Game24Evaluator(
        output_extractor=utils.parse_response,
        answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x),
        disable_log=False,
        disable_tqdm=False,
        sample_prompt_type="o1",
        prompt= prompt,
    )
    accuracy = evaluator.evaluate(
        reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir
    )
    print(f"accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
