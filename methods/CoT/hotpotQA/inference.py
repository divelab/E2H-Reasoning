import json
from typing import Literal

import fire
import utils

from reasoners.benchmark import HotpotQAEvaluator
from reasoners.lm.hf_model import HFModel
from reasoners.lm.llama_api_model import LLaMaApiModel
from reasoners.lm.openai_model import OpenAIModel


class CoTReasoner:
    def __init__(self, base_model, n_sc=1, temperature=0, bs=1):
        assert n_sc == 1 or temperature > 0, (
            "Temperature = 0 indicates greedy decoding. There is no point running multiple chains (n_sc > 1)"
        )
        self.base_model = base_model
        self.temperature = temperature
        self.n_sc = n_sc
        self.bs = bs

    def __call__(self, example, prompt=None):
        inputs = prompt["cot"].replace("{QUESTION}", example)
        outputs = []
        do_sample = True
        if self.temperature == 0 and isinstance(self.base_model, HFModel):
            print("Using greedy decoding with HF model. Set do_sample=False")
            self.temperature == 1.0
            do_sample = False
        if isinstance(self.base_model, OpenAIModel) or isinstance(
            self.base_model, LLaMaApiModel
        ):
            eos_token_id = []
        else:
            # Have to manually set for any new models
            eos_token_id = [13]
        for i in range((self.n_sc - 1) // self.bs + 1):
            local_bs = min(self.bs, self.n_sc - i * self.bs)
            outputs += self.base_model.generate(
                [inputs] * local_bs,
                hide_input=True,
                do_sample=do_sample,
                temperature=self.temperature,
                eos_token_id=eos_token_id,
            ).text
        outputs = [
            o.strip() if o.strip().endswith(".") else o.strip() + "." for o in outputs
        ]
        # print("OUTPUTS:", outputs)
        return outputs


def main(
    base_lm: Literal["hf", "openai", "api"],
    model_dir=None,
    api_model_id="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    openai_model="gpt-4o-mini",
    batch_size=1,
    prompt="prompts/hotpotQA/prompts.json",
    resume=0,
    log_dir=None,
    temperature=0,
    n_sc=5,
    quantized="int8",
):
    if base_lm == "openai":
        base_model = OpenAIModel(openai_model, additional_prompt="ANSWER")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir, quantized=quantized)
    elif base_lm == "api":
        base_model = LLaMaApiModel(
            None,
            None,
            use_api=True,
            model_id=api_model_id,
            quantized=None,
            additional_prompt="ANSWER",
        )
        model_dir = base_model.model_id
    else:
        raise ValueError(f"Unknown base_lm: {base_lm}")
    with open(prompt) as f:
        prompt = json.load(f)

    reasoner = CoTReasoner(
        base_model, temperature=temperature, n_sc=n_sc, bs=batch_size
    )
    evaluator = HotpotQAEvaluator(
        output_extractor=utils.cot_sc_extractor,
        answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
        init_prompt=prompt,
        disable_log=False,
        disable_tqdm=False,
        sample_prompt_type="cot",
    )
    from datetime import datetime

    log_dir = f"logs/hotpotQA/cot/{datetime.now().strftime('%m%d%Y-%H%M%S')}"
    if base_lm == "hf" or base_lm == "api":
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
