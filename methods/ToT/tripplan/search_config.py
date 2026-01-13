from reasoners import WorldModel, LanguageModel, SearchConfig
from world_model import TripPlanState, TripPlanAction
from typing import Type, Callable, Optional, Literal
import utils

class TPConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 temperature: float = 0.8,
                 n_candidate: int = 4,
                 calc_reward: Literal['sampling', 'logits'] = 'sampling') -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.n_candidate = n_candidate
        self.temperature = temperature
        self.calc_reward = calc_reward

    def get_actions(self, state: TripPlanState) -> list[TripPlanAction]:
        input_prompt = (self.prompt['tot_prefix'] + self.prompt['tot']).replace("{Question}", self.example)
        input_prompt += "".join([" " + s for s in state.state_history])
        print("Action history: ", "".join([" " + s for s in state.state_history]))

        outputs = self.base_model.generate([input_prompt],
                                          num_return_sequences=self.n_candidate,
                                          stop=".",
                                          temperature=self.temperature,
                                          do_sample=True,
                                          hide_input=True,
                                          additional_prompt="CONTINUE",
                                          ).text
        
        outputs = [o + '.' for o in outputs]
        print("-------------------------------")

        # deduplicate
        outputs = list(dict.fromkeys(outputs))
        print("ACTIONS Outputs: with temperature  ", self.temperature,"  :  ", outputs)
        actions = []
        for output in outputs:
          # get stay duration
          start_day = utils.calculate_start_day(output)
          end_day = utils.calculate_end_day(output)
          if end_day == None or start_day < state.current_day:
             continue
          actions.append(TripPlanAction(current_day=end_day, action=output))
        
        if len(actions) == 0:
           output = self.base_model.generate([input_prompt], temperature=self.temperature, do_sample=True, hide_input=True, additional_prompt="CONTINUE").text
           new_actions = []
           new_actions.append(TripPlanAction(current_day=200, action=output[0]))
           return new_actions
        
        return actions


    def fast_reward(self, state: TripPlanState, action: TripPlanAction) -> tuple[float, dict]:
        if self.calc_reward == 'sampling':
          input_prompt = (self.prompt['tot_prefix'] + self.prompt['tot']).replace("{Question}", self.example)
          input_prompt += "".join([" " + s for s in state.state_history])

          rating_prompt = f"Given the prompt:\n{input_prompt}\n\n" \
                          f"Rate the action:\n'{action.action}'\non a scale from 1 to 10. Do not grade easy. Provide only a number in response."
          rating_response = self.base_model.generate([rating_prompt])
          try:
            rating = float(rating_response.text[0].strip())
          except:
            rating = 0
          return rating, {'intuition': rating, 'self_eval': rating}
        
        inputs = (self.prompt['tot_prefix'] + self.prompt['tot']).replace("{Question}", self.example)
        inputs += "".join([" " + s for s in state.state_history])
        intuition = self.base_model.get_loglikelihood(inputs + "\n", [inputs + "\n" + action.action])[0]

        self_eval_prompt = self.prompt['tot_prefix'] + self.prompt["self-eval"].replace("{Question}", self.example)
        self_eval_prompt += "\nACTION:\n" + action.action + "\nEVALUATION:\n"
        self_eval = self.base_model.get_loglikelihood(self_eval_prompt, 
            [self_eval_prompt + "good"])[0]
        
        print('Fast Reward Logits', self_eval, intuition)
        return intuition + self_eval, {'intuition': intuition, "self_eval": self_eval}


    def reward(self, state: TripPlanState, action: TripPlanAction, **kwargs) -> tuple[float, dict]:
        intuition, self_eval = kwargs['intuition'], kwargs['self_eval']
        return self_eval, {'intuition': intuition, "self_eval": self_eval}

    def update_example(self, example, prompt=None) -> None:
        super().update_example(example, prompt=prompt)