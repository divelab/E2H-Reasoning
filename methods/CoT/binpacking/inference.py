from datetime import datetime
import json
import fire
from typing import Literal
import json
import transformers
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.hf_model import HFModel
from reasoners.lm.llama_api_model import LLaMaApiModel
from reasoners.benchmark.binpacking import BinPackingEvaluator
import utils
from tqdm import tqdm

class CoTReasoner():
    def __init__(self, base_model, temperature=0.8, sc_num = 1):
        self.base_model = base_model
        self.temperature = temperature
        self.sc_num = sc_num
        
    def __call__(self, example, prompt=None):
        inputs = prompt["cot"].replace("{item_sizes}", str(example['item_sizes'])).replace("{bin_size}", str(example['bin_capacity']))
        outputs = []
        do_sample = True
        # print(inputs)
        if self.temperature == 0 and isinstance(self.base_model, HFModel):
            print("Using greedy decoding with HF model. Set do_sample=False")
            self.temperature == 1.0
            do_sample = False
        if isinstance(self.base_model, OpenAIModel) or isinstance(self.base_model, LLaMaApiModel):
            eos_token_id = []
        else:
            print(self.base_model.model.__class__)
            print(self.base_model.model.config.architectures[0])
            assert isinstance(self.base_model.model, transformers.LlamaForCausalLM)
            eos_token_id = [13]
        
        for _ in tqdm(range(self.sc_num), leave=False):
            
            output = self.base_model.generate([inputs],
                                          hide_input=True,
                                          do_sample=do_sample,
                                          temperature=self.temperature,
                                          eos_token_id=eos_token_id).text[0].strip()
            outputs.append(output)
        return outputs

def main(base_lm:Literal['hf', 'openai', 'api'] = 'openai',
         model_dir= None, 
         prompt_path="prompts/binpacking/prompts.json", 
         quantized='int8',
         resume=0, 
         temperature=0.8,
         sc_num=1,
         openai_model='gpt-4o-mini',
         api_model_id='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
         log_dir=None):

    if base_lm == "openai":
        base_model = OpenAIModel(openai_model, additional_prompt="CONTINUE")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir,quantized=quantized, additional_prompt="CONTINUE")
    elif base_lm == 'api':
        base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None, additional_prompt="CONTINUE")
        model_dir = base_model.model_id
    else:
        raise ValueError(f"base_lm {base_lm} is not supported")
    
    if base_lm == 'hf' or base_lm == 'api':
        model_name= model_dir.split('/')[-1]
    else:
        model_name = base_lm
    
    # Load testset with initial config, otherwise prompt model to generate it first.
    with open(prompt_path) as f:
        prompt = json.load(f) 
        
    log_dir =  f'logs/binpacking/'\
                        f'CoT/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    
    # logs storage
    log_dir = log_dir + f'_{model_name}'
    
    reasoner = CoTReasoner(base_model, temperature=temperature, sc_num=sc_num)
    evaluator = BinPackingEvaluator(
        output_extractor=utils.cot_sc_extractor,
        answer_extractor=lambda x: x,
        init_prompt=prompt, # will update dynamically
        disable_log=False,
        disable_tqdm=False,
        sample_prompt_type="cot"
    )
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=5, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    
    

if __name__ == '__main__':
    fire.Fire(main)