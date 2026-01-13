import json
from datetime import datetime
from typing import Literal

import fire
import utils

from reasoners.benchmark import CalendarPlanEvaluator
from reasoners.lm.hf_model import HFModel
from reasoners.lm.openai_model import OpenAIModel


class IOReasoner:
    def __init__(self, base_model, temperature=0.8):
        self.base_model = base_model
        self.temperature = temperature

    def __call__(self, example, prompt=None):
        inputs = prompt["o1"].replace("{Question}", example)

        do_sample = True
        if isinstance(self.base_model, OpenAIModel):
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
    base_lm: Literal["hf", "openai"],
    model_dir=None,
    openai_model="o1-mini",
    num_days=None,
    data_path="data/calendarplan/output_num_days_{num}.json",
    prompt="prompts/calendarplan/prompts.json",
    quantized="int8",
    resume=0,
    temperature=0,
    log_dir=None,
):
    if base_lm == "openai":
        base_model = OpenAIModel(openai_model, additional_prompt="NONE")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir, quantized=quantized)
    else:
        raise ValueError(f"base_lm {base_lm} is not supported")

    if base_lm == "hf":
        model_name = model_dir.split("/")[-1]
    else:
        model_name = f"{base_lm}_{base_model.model}"

    data_path = data_path.format(num=f"{num_days}")

    with open(prompt) as f:
        prompt = json.load(f)

    log_dir = (
        f"logs/calendarplan/num_days-{num_days}/"
        f"CoT/"
        f"{datetime.now().strftime('%m%d%Y-%H%M%S')}"
    )

    log_dir = log_dir + f"_{model_name}"

    reasoner = IOReasoner(base_model, temperature=temperature)
    evaluator = CalendarPlanEvaluator(
        output_extractor=utils.parse_response,
        answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x),
        dataset_path=data_path,
        init_prompt=prompt,
        disable_log=False,
        disable_tqdm=False,
        sample_prompt_type="o1",
        prompt_key="prompt_0shot",
    )
    accuracy = evaluator.evaluate(
        reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir
    )
    print(f"accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
