import json
import fire
from typing import Sequence, Any
import json

from dataset import ProntoQADataset
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm.hf_model import HFModel
from reasoners.lm.gemini_model import BardCompletionModel
from reasoners.lm.anthropic_model import ClaudeModel
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
    
class IOReasoner:
    def __init__(self, base_model, temperature=0.8):
        self.base_model = base_model
        self.temperature = temperature

    def __call__(self, example, prompt=None):
        inputs = prompt
        inputs += "Q: " + example.test_example.question + " " + example.test_example.query + "\nA:"
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
        print(output)
        steps = [s.split("So")[1].strip()+'.' for s in output.split('.') if "So" in s]
        parsed_output = "\n".join(steps)
        print("Parsed output", parsed_output)
        return [parsed_output]

def main(base_lm='openai', model_dir=None, temperature=0.8, log_dir="name", quantized="int8", api_model_id='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', **kwargs):

    if base_lm == "openai":
        language_model = OpenAIModel("o1", additional_prompt="CONTINUE")
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
                        f'/o1/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}_{model_name}'

    with open('prompts/prontoqa/example_next_steps.json') as f:
        init_prompt = json.load(f)
    
    reasoner =  IOReasoner(base_model=language_model, temperature=temperature)

    evaluator = ProntoQAEvaluatorFinal(
        init_prompt=init_prompt['next_steps'],
        sample_prompt_type="o1",
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