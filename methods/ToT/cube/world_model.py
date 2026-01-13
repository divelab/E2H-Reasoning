import copy
from typing import List, NamedTuple

from reasoners import LanguageModel, WorldModel


class CubeState(NamedTuple):
    state_history: List[str]
    last_action: str


CubeAction = str


class CubeWorldModel(WorldModel):
    def __init__(self, base_model: LanguageModel, prompt: str) -> None:
        self.base_model = base_model
        self.prompt = prompt

    def init_state(self) -> CubeState:
        return CubeState([], "")

    def step(self, state: CubeState, action: CubeAction) -> tuple[CubeState, dict]:
        """Take a step in the world model.

        :param state: the current state (see the docstring of WorldModel)
        :param action: the action to take
        :return: the next state and additional information cached for reward calculation
        """
        new_state_history = copy.deepcopy(state.state_history)
        new_state_history.append(action)

        return CubeState(state_history=new_state_history, last_action=action), {}

    def is_terminal(self, state: CubeState):
        last_action = state.last_action.lower()
        print("Action history: ", state.state_history)
        print("Last action: ", state.last_action)
        return "cube is solved" in last_action

