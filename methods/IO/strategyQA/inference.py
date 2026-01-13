import json
import fire
from reasoners.lm import HFModel
from reasoners.lm.anthropic_model import ClaudeModel
from typing import Literal
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.gemini_model import BardCompletionModel
from datetime import datetime
from reasoners.benchmark import StrategyQAEvaluator
from utils import extract_final_answer, eval_output, extract_answer, retrieve_answer_from_dataset

class IOReasoner:
    def __init__(self, base_model, temperature=0.8):
        self.base_model = base_model
        self.temperature = temperature

    def __call__(self, example, prompt=None):
        prompt = prompt["o1"]
        question = example
        inputs = f"{prompt}\n\nQ: {question}\nA:"
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
    
def main(base_lm: Literal['hf', 'openai'] = 'openai',
            model_dir: str = None,
            prompt: str = 'prompts/strategyqa/prompts.json',
            data_path: str = 'data/strategyqa/strategyqa_test.json',
            resume: int = 0,
            log_dir: str = None,
            temperature: float = 0.8,
            quantized = 'int8',
            **search_algo_params):

    if base_lm == 'hf':
        base_model = HFModel(model_dir, model_dir,quantized=quantized)
    elif base_lm == 'openai':
        base_model = OpenAIModel("o1", additional_prompt="ANSWER")
    with open(prompt) as f:
        prompt = json.load(f)

    log_dir =  f'logs/strategyqa'\
                        f'/o1/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    if base_lm == 'hf':
        model_name = model_dir.split('/')[-1]
    else:
        model_name = base_lm
    log_dir = log_dir + f'_{model_name}'

    reasoner = IOReasoner(base_model=base_model, temperature=temperature)
    evaluator = StrategyQAEvaluator(
                 output_extractor= extract_answer,
                 answer_extractor=lambda x: retrieve_answer_from_dataset(x["answer"]),
                 init_prompt=prompt, # will update dynamically
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="o1",
                 dataset_path=data_path)
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    
    

def eval_output(answer, output):
    if output is None:
        return False
    
    # False vs no and True vs yes
    answer = "no" if not answer else "yes"
    
    return answer == output.strip().lower()

if __name__ == '__main__':
    fire.Fire(main)