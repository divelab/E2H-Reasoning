from reasoners import WorldModel, LanguageModel, SearchConfig
from world_model import GSM8KState, GSM8KAction
import utils
from typing import Literal

class GSM8KConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 temperature: float = 0.8,
                 n_candidate: int = 3,
                 calc_reward: Literal['sampling', 'logits'] = 'sampling',
                 depth_limit = 10) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.n_candidate = n_candidate
        self.temperature = temperature
        self.calc_reward = calc_reward
        self.depth_limit = depth_limit
       
    def get_actions(self, state: GSM8KState) -> list[GSM8KAction]:
        input_prompt = self.prompt['cot-context'] + self.prompt['tot']
        input_prompt = input_prompt.replace("{QUESTION}", self.example)
        input_prompt += "".join(["\n" + s for s in state.state_history])
        print("Action history: ", "".join(["  " + s for s in state.state_history]))

        outputs = []
        if len(state.state_history) == self.depth_limit - 1:
            input_prompt += self.prompt["answer-prompt"]

        outputs = self.base_model.generate([input_prompt],
                              num_return_sequences=self.n_candidate,
                              stop="\n",
                              temperature=self.temperature,
                              do_sample=True,
                              hide_input=True,
                              use_api=True,
                              additional_prompt="CONTINUE").text

        # Filter outputs here
        print("-------------------------------")
        ret = [o.strip() for o in outputs if o.strip() not in state.state_history and o != "So the answer is"]

        if len(ret) == 0:
            input_prompt += self.prompt["answer-prompt"]
            output = self.base_model.generate([input_prompt], temperature=self.temperature, use_api=True).text
            print(output)
            return output
        
        # deduplicate
        ret = list(dict.fromkeys(ret))
        print("ACTIONS Outputs: with temperature  ", self.temperature,"  :  ", ret)
        return ret


    def fast_reward(self, state: GSM8KState, action: GSM8KAction) -> tuple[float, dict]:
        input_prompt = (self.prompt['cot-context'] + self.prompt['tot']).replace("{QUESTION}", self.example) + "".join(["\n" + s for s in state.state_history])
        if self.calc_reward == "sampling":
          rating_prompt = self.prompt["reward-prompt"].replace("{input_prompt}", input_prompt).replace("{action}", action)
          rating_response = self.base_model.generate([rating_prompt])
          try:
            rating = float(rating_response.text[0].strip())
          except:
            rating = 0
          return rating, {'intuition': rating, 'self_eval': rating}

        intuition = self.base_model.get_loglikelihood(input_prompt + "\n", [input_prompt + "\n" + action])[0]
        self_eval_prompt = self.prompt['cot-context'] + self.prompt["self-eval"].replace("{Question}", self.example)
        self_eval_prompt += "\nACTION:\n" + action + "\nEVALUATION:\n"
        self_eval = self.base_model.get_loglikelihood(self_eval_prompt, 
            [self_eval_prompt + "good"])[0]
        
        print('Fast Reward Logits', self_eval, intuition)
        return intuition + self_eval, {'intuition': intuition, "self_eval": self_eval}

    def reward(self, state: GSM8KState, action: GSM8KAction, **kwargs) -> tuple[float, dict]:
        intuition, self_eval = kwargs['intuition'], kwargs['self_eval']
        return self_eval, {'intuition': intuition, "self_eval": self_eval}

    def update_example(self, example, prompt=None) -> None:
        super().update_example(example, prompt=prompt)