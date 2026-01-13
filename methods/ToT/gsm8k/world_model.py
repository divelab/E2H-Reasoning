from typing import List, Dict, Tuple, NamedTuple
from reasoners import WorldModel, LanguageModel
import json
import copy 

class GSM8KState(NamedTuple):
    state_history: List[str]
    last_action: str

GSM8KAction = str

class GSM8KWorldModel(WorldModel):
    def __init__(self, base_model: LanguageModel, prompt: str) -> None:
        self.base_model = base_model
        self.prompt = prompt
    
    def init_state(self) -> GSM8KState:
        return GSM8KState([], "")

    def step(self, state: GSM8KState, action: GSM8KAction) -> tuple[GSM8KState, dict]:
        """Take a step in the world model.
        
        :param state: the current state (see the docstring of WorldModel)
        :param action: the action to take
        :return: the next state and additional information cached for reward calculation
        """
        new_state_history = copy.deepcopy(state.state_history)
        new_state_history.append(action)

        return GSM8KState(state_history=new_state_history, last_action=action), {}


    def is_terminal(self, state: GSM8KState):
      # last action starts with The answer is
      last_action = state.last_action.replace("So the", "The")
      return last_action.startswith("The answer is")