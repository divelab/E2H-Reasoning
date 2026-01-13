from reasoners import WorldModel, LanguageModel, SearchConfig
from typing import NamedTuple
import copy

BWAction = str
class BWState(NamedTuple):
    """The state of the Blocksworld for ToT

    See the docstring of BlocksWorldModel for more details.
    """
    step_idx: int
    action_history: list[str]
    end: bool



class BlocksWorldModel(WorldModel):
    """Blocks World World Model
    State: (step_idx, action_history: [str])
    Action: e.g. "put the red block on the green block"
    """

    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 max_steps: int = 6,
                 batch_size=1) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt  # need to check if this is necessary
        self.batch_size = batch_size

    def init_state(self) -> BWState:
        """Initialize the world model.

        :return: the initial state
        """
        return BWState(step_idx=0, action_history=[], end=False)

    def step(self, state: BWState, action: BWAction) -> tuple[BWState, dict]:
        """Take a step in the world model.
        
        :param state: the current state (see the docstring of BlocksWorldModel)
        :param action: the action to take
        :return: the next state and additional information cached for reward calculation
        """
        state = copy.deepcopy(state)
        if action != "[PLAN END]":
            state = BWState(step_idx=state.step_idx + 1, action_history=state.action_history + [action], end=False)
        else:
            state = BWState(step_idx=state.step_idx + 1, action_history=state.action_history, end=True)
        return state, {}

    def is_terminal(self, state: BWState) -> bool:
        if state.end:
            return True
        elif state.step_idx == self.max_steps:
            return True
        return False