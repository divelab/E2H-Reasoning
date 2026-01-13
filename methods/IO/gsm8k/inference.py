import json
from reasoners.lm.openai_model import OpenAIModel
from reasoners.benchmark import GSM8KEvaluator
from reasoners.lm.hf_model import HFModel
from reasoners.lm.gemini_model import BardCompletionModel
from reasoners.lm.anthropic_model import ClaudeModel
from reasoners.lm.llama_api_model import LLaMaApiModel
import utils
from typing import Literal
import fire

class IOReasoner:
    def __init__(self, base_model, temperature=0.8):
        self.base_model = base_model
        self.temperature = temperature

    def __call__(self, example, prompt=None):
        inputs = prompt["o1"].replace("{QUESTION}", example)

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
        output += "."
        return [output]

def main(base_lm:Literal['hf', 'openai', "api"],
         model_dir=None,
         prompt="prompts/gsm8k/prompts.json", 
         resume=0, log_dir=None, temperature=0.8, quantized='int8',
         api_model_id='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
         openai_model="o1"):

    if base_lm == "openai":
        base_model = OpenAIModel(openai_model, additional_prompt="ANSWER")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir, quantized=quantized)
    elif base_lm == 'api':
        base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None, additional_prompt="ANSWER")
        model_dir = base_model.model_id
    else:
        raise ValueError(f"Unknown base_lm: {base_lm}")
    with open(prompt) as f:
        prompt = json.load(f)

    reasoner = IOReasoner(base_model, temperature=temperature)
    evaluator = GSM8KEvaluator(
                 output_extractor=utils.cot_sc_extractor,
                 answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
                 init_prompt=prompt,
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="o1")
    
    from datetime import datetime
    log_dir =  f'logs/gsm8k/'\
                        f'/o1/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    if base_lm == 'hf' or base_lm == 'api':
        model_name= model_dir.split('/')[-1]
    else:
        model_name = base_lm
    log_dir = log_dir + f'_{model_name}'
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    return 0

if __name__ == '__main__':
    fire.Fire(main)
"""
CUDA_VISIBLE_DEVICES=2 python examples/cot/cot_gsm8k/inference.py \
--model_dir $Gemma_ckpts \ 
"""


