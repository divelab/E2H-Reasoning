import re
from reasoners.lm import HFModel
from datetime import datetime
import json
from reasoners.benchmark import BWEvaluator
import fire
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.llama_api_model import LLaMaApiModel

def extract_step(file_path):
    match = re.search(r'_step_(\d+)_', file_path)
    return int(match.group(1)) if match else None

def sc_output_extractor(algo_output):
    from collections import Counter
    answers = [x for x in algo_output if x is not None]
    counter = Counter(answers)
    if counter == {}:
        return None
    return counter.most_common(1)[0][0]

class CoTReasoner():
    def __init__(self, base_model, temperature=0.8, sc_num = 1, model_type="completion"):
        self.base_model = base_model
        self.temperature = temperature
        self.model_type = model_type
        self.sc_num = sc_num

    def __call__(self, example, prompt=None):
        inputs = prompt["icl"].replace("<init_state>", example["init"])\
            .replace("<goals>", example["goal"]).replace("<action>", "")

        outputs = []
        for _ in range(self.sc_num):
          if self.model_type == "completion":
              outputs.append(self.base_model.generate([inputs],
                                            hide_input=True,
                                            do_sample=True,
                                            temperature=self.temperature,
                                            eos_token_id='\n[').text[0][:-1].strip())
          elif self.model_type == "chat":
              outputs.append(self.base_model.generate([inputs],
                                            hide_input=True,
                                            do_sample=True,
                                            temperature=self.temperature).text[0].replace("[PLAN END]", "").strip()) 
            
        return outputs

def main(base_lm, prompt_path='prompts/blocksworld/pool_prompt_v1.json', 
         model_dir = None, 
         disable_log=False, 
         config_file: str = "data/blocksworld/bw_config.yaml", 
         domain_file: str = "data/blocksworld/generated_domain.pddl", 
         resume=0, 
         log_dir=None,
         temperature=0.8,
         quantized="int8",
         sc_num=1,
         data_path='data/blocksworld/split_v1/split_v1_step_2_data.json',
         openai_model="gpt-4o-mini",
         api_model_id='meta-llama/Meta-Llama-3.1-8B-Instruct'):

    if base_lm == "openai":
        base_model = OpenAIModel(openai_model, additional_prompt="CONTINUE")
    elif base_lm == "hf":
        base_model = HFModel(model_pth=model_dir, tokenizer_pth=model_dir, quantized=quantized)
    elif base_lm == "api":
        base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None, additional_prompt="CONTINUE")
        model_dir = base_model.model_id
    with open(prompt_path) as f:
        prompt = json.load(f)

    print("Quantized: ", quantized)

    if base_lm == 'hf' or base_lm == 'api':
        model_name= model_dir.split('/')[-1]
    else:
        model_name = base_lm

    log_dir =  f'logs/Blocksworld/'\
                        f'CoT/step_{extract_step(data_path)}/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    log_dir = log_dir + f'_{model_name}'

    reasoner = CoTReasoner(base_model, temperature=temperature, sc_num=sc_num)
    evaluator = BWEvaluator(config_file=config_file, domain_file=domain_file, data_path=data_path, init_prompt=prompt, disable_log=disable_log, output_extractor=sc_output_extractor, sample_prompt_type="rap") # rap prompt includes cot
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    return 0

if __name__ == '__main__':
    fire.Fire(main)