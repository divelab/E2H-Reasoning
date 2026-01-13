import json
import fire
from typing import Sequence, Any
import json
from data.prontoqa.dataset import ProntoQADataset
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.hf_model import HFModel
from reasoners.benchmark import ProntoQAEvaluatorFinal
from reasoners.lm.llama_api_model import LLaMaApiModel
from datetime import datetime

def sc_output_extractor(algo_output):
    from collections import Counter
    answers = [x for x in algo_output if x is not None]
    counter = Counter(answers)
    if counter == {}:
        return None
    return counter.most_common(1)[0][0]

class CoTReasoner():

    def __init__(self, base_model, n_sc=1, temperature=0.8, bs=1):
        assert n_sc == 1 or temperature > 0, \
            "Temperature = 0 indicates greedy decoding. There is no point running multiple chains (n_sc > 1)"
        self.base_model = base_model
        self.temperature = temperature
        self.n_sc = n_sc
        self.bs = bs


    def __call__(self, example, prompt=None):
        input_prompt = prompt
        input_prompt += "Q: " + example.test_example.question + " " + example.test_example.query + "\nA:"
        print(f"input_prompt: '{input_prompt}'\n")

        if isinstance(self.base_model, OpenAIModel) or isinstance(self.base_model, LLaMaApiModel):
            eos_token_id = []
        else:
            eos_token_id = [13]

        outputs = []
        for _ in range(self.n_sc):
          output = self.base_model.generate([input_prompt], eos_token_id=eos_token_id, hide_input=True, temperature=self.temperature, do_sample=True).text[0]
          print(f"output: '{output}'\n")
          steps = [s.split("So")[1].strip()+'.' for s in output.split('.') if "So" in s]
          outputs.append("\n".join(steps))

        return outputs

def main(base_lm='openai', 
         model_dir=None, 
         temperature=0.8, 
         quantized="int8", 
         sc_num=1, 
         openai_model="gpt-4o-mini",
         api_model_id='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'):

    if base_lm == "openai":
        language_model = OpenAIModel(openai_model, additional_prompt="CONTINUE")
    elif base_lm == "hf":
        language_model = HFModel(model_pth=model_dir, tokenizer_pth=model_dir, quantized=quantized)
    elif base_lm == "api":
        language_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None, additional_prompt="CONTINUE")
        model_dir = language_model.model_id
    else:
        raise ValueError(f"Unknown model: {base_lm}")
    
    if base_lm == 'hf' or base_lm == 'api':
        model_name= model_dir.split('/')[-1]
    else:
        model_name = f'{base_lm}_{language_model.model}'   
    
    log_dir =  f'logs/prontoqa'\
                        f'/cot/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}_{model_name}'

    with open('prompts/prontoqa/example_next_steps.json') as f:
        init_prompt = json.load(f)
    
    reasoner =  CoTReasoner(base_model=language_model, temperature=temperature, n_sc=sc_num)
    evaluator = ProntoQAEvaluatorFinal(
        init_prompt=init_prompt['next_steps'],
        sample_prompt_type="cot",
        disable_log=False,
        disable_tqdm=False, dataset = ProntoQADataset.from_file(
            'data/prontoqa/345hop_random_true.json'
        ),
        output_extractor=sc_output_extractor,
        answer_extractor=lambda x: "\n".join(x.test_example.chain_of_thought[2::2])
    )

    accuracy = evaluator.evaluate(reasoner, num_shot=4, log_dir=log_dir)
    print(f"accuracy: {accuracy}")

if __name__ == '__main__':
    fire.Fire(main)