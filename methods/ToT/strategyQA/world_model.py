from typing import List, NamedTuple
from reasoners import WorldModel, LanguageModel
import copy 

class SQAState(NamedTuple):
    state_history: List[str]
    last_action: str

SQAAction = str

class SQAWorldModel(WorldModel):
    def __init__(self, base_model: LanguageModel, prompt: str) -> None:
        self.base_model = base_model
        self.prompt = prompt
    
    def init_state(self) -> SQAState:
        return SQAState([], "")

    def step(self, state: SQAState, action: SQAAction) -> tuple[SQAState, dict]:
        """Take a step in the world model.
        
        :param state: the current state (see the docstring of WorldModel)
        :param action: the action to take
        :return: the next state and additional information cached for reward calculation
        """
        new_state_history = copy.deepcopy(state.state_history)
        new_state_history.append(action)

        return SQAState(state_history=new_state_history, last_action=action), {}


    def is_terminal(self, state: SQAState):
      # last action contains "The answer is"
      last_action = state.last_action
      return "The answer is" in last_action or "the answer is" in last_action