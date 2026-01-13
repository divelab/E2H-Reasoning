import sys
import torch 

from reasoners import SearchConfig
from world_model import ProntoQAState, ProntoQAAction, ProntoQAExample
from reasoners import SearchConfig
from typing import Literal
import prompts.prontoqa.finish
import prompts.prontoqa.next_step
import prompts.prontoqa.valid_tot

def remove_so_prefix(s):
    if s.startswith('So '):
        return s[3:]
    return s

class ProntoQAToTSearchConfig(SearchConfig[ProntoQAState, ProntoQAAction, ProntoQAExample]):
    def __init__(self, base_model, n_actions=5, temperature=0.8, calc_reward: Literal['sampling', 'logits'] = 'sampling') -> None:
        super().__init__()
        self.n_actions = n_actions
        self.temperature = temperature
        self.base_model = base_model
        self.calc_reward = calc_reward
        assert temperature > 0, "Temperature = 0 indicates greedy decoding. There is no point running multiple chains"
        
    def get_actions(self, state: ProntoQAState) -> list[ProntoQAAction]:
        input_prompt = self.prompt
        input_prompt += "Q: " + self.example.test_example.question + " " + self.example.test_example.query + "\nA:"
        input_prompt += "".join([" " + s for s in state])

        output = self.base_model.generate([input_prompt],
                                          num_return_sequences=self.n_actions,
                                          temperature=0.8, #self.temperature,
                                          stop=".",
                                          do_sample=True,
                                          additional_prompt="CONTINUE",
                                          hide_input=True).text

        ret = [o.strip() + '.' for o in output]
        print(f"model generated actions: {ret}")
        # deduplicate
        ret = dict.fromkeys(ret).keys()
        print("Deduplicated actions: ", ret)
        return ret

    def fast_reward(self, state: ProntoQAState, action: ProntoQAAction) -> tuple[float, dict]:
        processed_state = [remove_so_prefix(s) for s in state]
        processed_action = remove_so_prefix(action)
        input_prompt = self.prompt
        input_prompt += "Q: " + self.example.test_example.question + " " + self.example.test_example.query + "\nA:"
        input_prompt += "".join([" " + s for s in processed_state])
        candidate = input_prompt + " " + processed_action
        if self.calc_reward == 'sampling':
            rating_prompt = f"Given the prompt:\n{input_prompt}\n" \
            f"Rate the action:\n'{processed_action}'\non a scale from 1 to 10. Do not grade easy. Provide only a number in response."

            rating_response = self.base_model.generate([rating_prompt])
            try:
              rating = float(rating_response.text[0].strip())
            except:
              rating = 0
            print("RATING: ", rating)
            return rating, {'intuition': rating, 'self_eval': rating}

        intuition = self.base_model.get_loglikelihood(input_prompt, 
            [candidate])[0]
        print(f" prompt: {self.prompt}")
        print(f"action: {processed_action}")
        print(f"input_prompt: {input_prompt}")
        print(f"state: {processed_state}")

        input_prompt = ""
        input_prompt += prompts.valid_tot.EXAMPLES
        input_prompt += prompts.valid_tot.FACTS_FORMAT.format(self.example.test_example.question or "", self.example.test_example.query)
        input_prompt += prompts.valid_tot.NEXT_STEP_FORMAT.format(',\n'.join(f'"{statement}"' for statement in processed_state))
        input_prompt += prompts.valid_tot.VALID_PREFIX

        output_logits = self.base_model.get_next_token_logits(
            input_prompt,
            candidates=["Yes", "No"]
        )

        print(f"input_prompt: {input_prompt}")
        reward: float = output_logits[0][0].item()
        reward:float = torch.softmax(torch.tensor(output_logits[0]), dim=0)[0].item()
        print(f" reward: {reward}")

        self_eval = reward  
        print(f" intuition: {intuition}, self_eval: {self_eval}")
        return intuition*0.5 + self_eval*0.5, {"intuition": intuition, "self_eval":self_eval}

    def reward(self, state, action, **kwargs) -> tuple[float, dict]:
        # how correct is this last action
        intuition = kwargs["intuition"]
        self_eval = kwargs["self_eval"]
        return intuition*0.5 + self_eval*0.5, {"intuition": intuition, "self_eval":self_eval}
