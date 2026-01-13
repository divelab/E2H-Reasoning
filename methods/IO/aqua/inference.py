from typing import Literal
import json
from reasoners.benchmark import AQuAEvaluator
import utils
import fire
from reasoners.lm.hf_model import HFModel
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.llama_api_model import LLaMaApiModel
from datetime import datetime
            
class IOReasoner:
    def __init__(self, base_model, temperature=0.8):
        self.base_model = base_model
        self.temperature = temperature

    def __call__(self, example, prompt=None):
        inputs = prompt["o1"].replace("{QUESTION}", example)

        do_sample = True
        if (
            isinstance(self.base_model, OpenAIModel)
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
        return [output]

def main(base_lm:Literal['hf', 'openai', 'api'],
         model_dir= None, 
         prompt="prompts/aqua/prompts.json", 
         data_path="data/aqua", 
         datasetname="test",
         quantized='int8',
         resume=0, 
         temperature=0,
         sc_num=1,
         api_model_id='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
         openai_model="o1-mini",
         log_dir=None):

    if base_lm == "openai":
        base_model = OpenAIModel(openai_model, additional_prompt="ANSWER")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir,quantized=quantized)
    elif base_lm == 'api':
        base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None, additional_prompt="ANSWER")
        model_dir = base_model.model_id
    else:
        raise ValueError(f"base_lm {base_lm} is not supported")
    with open(prompt) as f:
        prompt = json.load(f)

    reasoner = IOReasoner(base_model, temperature=temperature)
    
    output_extractor = utils.retrieve_answer
    if sc_num > 1:
        output_extractor = utils.cot_sc_extractor
    
    evaluator = AQuAEvaluator(
                 output_extractor=output_extractor,
                 answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
                 init_prompt=prompt, # will update dynamically
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="o1",
                 dataset_path=data_path,
                 datasetname=datasetname)
    
    log_dir =  f'logs/AQuA/'\
                        f'o1/'\
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