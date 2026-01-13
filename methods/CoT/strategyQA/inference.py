import json
import fire
from reasoners.lm import HFModel
from typing import Literal
from utils import extract_final_answer
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.llama_api_model import LLaMaApiModel
import transformers
from datetime import datetime
from reasoners.benchmark import StrategyQAEvaluator


class CotReasoner:
    def __init__(self, base_model, temperature=0.8, sc_num = 1):
        self.base_model = base_model
        self.temperature = temperature
        self.sc_num = sc_num
    
    def __call__(self, example, prompt=None):
        
        do_sample = True
        if self.temperature == 0 and isinstance(self.base_model, HFModel):
            print("Using greedy decoding with HF model. Set do_sample=False")
            self.temperature == 1.0
            do_sample = False
        
        input_prompt = f"{prompt['cot']}\n\nQ: {example}\nA:"

        print(input_prompt, flush=True)
        if isinstance(self.base_model, OpenAIModel) or isinstance(self.base_model, LLaMaApiModel):
            eos_token_id = []
        elif isinstance(self.base_model.model, transformers.GemmaForCausalLM):
            eos_token_id = [108,109]
        elif isinstance(self.base_model.model, transformers.MistralForCausalLM) or isinstance(self.base_model.model, transformers.MixtralForCausalLM):
            eos_token_id = [13]
        else:
            assert isinstance(self.base_model.model, transformers.LlamaForCausalLM)
            eos_token_id = [13]
        answers = []

        
        for _ in range(self.sc_num):
            # generate the answer
            output = self.base_model.generate(
                [input_prompt],
                do_sample=do_sample,
                temperature=self.temperature,
                num_return_sequences=1,
                eos_token_id=eos_token_id
            ).text[0]
            output_ans = extract_final_answer(output)
            answers.append(output_ans)
        
        if len(answers) == 0:
            final_answer = ""
        else:
            final_answer = max(answers, key=answers.count)
        
        return final_answer
        


def main(base_lm: Literal['hf','openai', 'api'] = 'openai',
            model_dir: str = None,
            sc_num: int = 1,
            prompt_path: str = 'prompts/strategyqa/prompts.json',
            data_path: str = 'data/strategyqa/strategyqa_test.json', 
            resume: int = 0,
            log_dir: str = None,
            temperature: float = 0,
            quantized = 'int8',
            openai_model='gpt-4o-mini',
            api_model_id='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
            **kwargs):


    if base_lm == 'hf':
        base_model = HFModel(model_dir, model_dir,quantized=quantized)
    elif base_lm == 'openai':
        base_model = OpenAIModel(openai_model, additional_prompt="ANSWER")
    elif base_lm == 'api':
        base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None)
        model_dir = base_model.model_id
    else:
        raise ValueError(f"base_lm {base_lm} is not supported")

    log_dir =  f'logs/strategyQA'\
                        f'/cot/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    if base_lm == 'hf' or base_lm == 'api':
        model_name = model_dir.split('/')[-1]
    else:
        model_name = base_lm
    log_dir = log_dir + f'_{model_name}'
    # load the dataset

    with open(prompt_path, 'r') as f:
        prompt = json.load(f)
    
    print("----------------")
        
    reasoner = CotReasoner(base_model, temperature=temperature, sc_num=sc_num)
    evaluator = StrategyQAEvaluator(
                 output_extractor= lambda x: x,
                 answer_extractor=lambda x: x["answer"],
                 init_prompt=prompt, # will update dynamically
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="cot",
                 resume=resume,
                 dataset_path=data_path)        
    
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=5, resume=resume, log_dir=log_dir)
    print(accuracy)


if __name__ == '__main__':
    fire.Fire(main)