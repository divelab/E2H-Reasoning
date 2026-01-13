from reasoners import WorldModel, LanguageModel, SearchConfig, State, Reasoner
from reasoners.algorithm import BeamSearch, MCTS
import reasoners.benchmark.bw_utils as utils
from typing import NamedTuple
import copy
import numpy as np

class CalendarStateTot(NamedTuple):
    step_idx: int
    action_history: list[str]
    end: bool

CalendarAction = str


class CalenderPlanWorldModel(WorldModel):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 max_steps: int = 2,
                 batch_size: int = 1) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size

    def init_state(self) -> CalendarStateTot:
        return CalendarStateTot(step_idx=0, action_history=[], end=False)
    
    def step(self, state: CalendarStateTot, action: CalendarAction) -> tuple[CalendarStateTot, dict]:
        state = copy.deepcopy(state)
        state = CalendarStateTot(
            step_idx=state.step_idx + 1,
            action_history=state.action_history + [action],
            end=state.step_idx + 1 >= self.max_steps  
        )
        return state, {}
    
    def is_terminal(self, state: State) -> bool:
        return state.end or state.step_idx >= self.max_steps

