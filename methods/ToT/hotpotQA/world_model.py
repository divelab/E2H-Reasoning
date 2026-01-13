import copy
import re
from typing import List, NamedTuple

from reasoners import LanguageModel, WorldModel


def get_first_sentence(text):
    match = re.search(r"(.+?[.!?])(\s|$)", text)
    if match:
        return match.group(1).strip()
    return text.strip()


class HotpotState(NamedTuple):
    state_history: List[str]
    last_action: str


HotpotAction = str


class HotpotWorldModel(WorldModel):
    def __init__(self, base_model: LanguageModel, prompt: str) -> None:
        self.base_model = base_model
        self.prompt = prompt

    def init_state(self) -> HotpotState:
        return HotpotState([], "")

    def step(
        self, state: HotpotState, action: HotpotAction
    ) -> tuple[HotpotState, dict]:
        """Take a step in the world model.

        :param state: the current state (see the docstring of WorldModel)
        :param action: the action to take
        :return: the next state and additional information cached for reward calculation
        """
        new_state_history = copy.deepcopy(state.state_history)
        action = get_first_sentence(action)
        new_state_history.append(action)

        return HotpotState(state_history=new_state_history, last_action=action), {}

    def is_terminal(self, state: HotpotState):
        # last action starts with The answer is
        last_action = state.last_action.replace("So the", "The")
        return last_action.startswith("The answer is")
