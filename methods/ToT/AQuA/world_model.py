from typing import List, Dict, Tuple, NamedTuple
from reasoners import WorldModel, LanguageModel
import json
import copy 

class AQUAState(NamedTuple):
    state_history: List[str]
    last_action: str

AQUAAction = str

class AQUAWorldModel(WorldModel):
    def __init__(self, base_model: LanguageModel, prompt: str) -> None:
        self.base_model = base_model
        self.prompt = prompt
    
    def init_state(self) -> AQUAState:
        return AQUAState([], "")

    def step(self, state: AQUAState, action: AQUAAction) -> tuple[AQUAState, dict]:
        """Take a step in the world model.
        
        :param state: the current state (see the docstring of WorldModel)
        :param action: the action to take
        :return: the next state and additional information cached for reward calculation
        """
        new_state_history = copy.deepcopy(state.state_history)
        new_state_history.append(action)

        return AQUAState(state_history=new_state_history, last_action=action), {}


    def is_terminal(self, state: AQUAState):
      # last action contains "The answer is"
      last_action = state.last_action.replace("So the", "The")
      return "The answer is" in last_action