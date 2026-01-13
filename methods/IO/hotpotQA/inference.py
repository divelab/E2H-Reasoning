import json
from typing import Literal

import fire
import utils

from reasoners.benchmark import HotpotQAEvaluator
from reasoners.lm.hf_model import HFModel
from reasoners.lm.openai_model import OpenAIModel


class IOReasoner:
    def __init__(self, base_model, n_sc=1, temperature=0, bs=1):
        assert n_sc == 1 or temperature > 0, (
            "Temperature = 0 indicates greedy decoding. There is no point running multiple chains (n_sc > 1)"
        )
        self.base_model = base_model
        self.temperature = temperature
        self.n_sc = n_sc
        self.bs = bs

    def __call__(self, example, prompt=None):
        inputs = prompt["o1"].replace("{QUESTION}", example)
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
        return [output]


def main(
    base_lm: Literal["hf", "openai"],
    model_dir,
    openai_model="o1-mini",
    batch_size=1,
    prompt="prompts/hotpotQA/prompts.json",
    resume=0,
    log_dir=None,
    temperature=0,
    n_sc=5,
    quantized="int8",
):
    if base_lm == "openai":
        base_model = OpenAIModel(openai_model, additional_prompt="NONE")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir, quantized=quantized)
    else:
        raise ValueError(f"Unknown base_lm: {base_lm}")
    with open(prompt) as f:
        prompt = json.load(f)

    reasoner = IOReasoner(base_model, temperature=temperature, n_sc=n_sc, bs=batch_size)
    evaluator = HotpotQAEvaluator(
        output_extractor=utils.cot_sc_extractor,
        answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
        init_prompt=prompt,  # will update dynamically
        disable_log=False,
        disable_tqdm=False,
        sample_prompt_type="o1",
    )
    from datetime import datetime

    log_dir = f"logs/hotpotQA/o1-mini/{datetime.now().strftime('%m%d%Y-%H%M%S')}"
    if base_lm == "hf":
        model_name = model_dir.split("/")[-1]
    else:
        model_name = base_lm
    log_dir = log_dir + f"_{model_name}"
    accuracy = evaluator.evaluate(
        reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir
    )
    print(f"accuracy: {accuracy:.4f}")
    return 0


if __name__ == "__main__":
    fire.Fire(main)
"""
CUDA_VISIBLE_DEVICES=2 python examples/cot/hotpotQA/inference.py \\
--model_dir $Gemma_ckpts \\
"""
