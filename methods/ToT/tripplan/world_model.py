from typing import List, Dict, Tuple, NamedTuple
from reasoners import WorldModel, LanguageModel
import json
import copy 

class TripPlanState(NamedTuple):
    current_day: int # The day you are currently in
    state_history: List[str]
    end: bool

class TripPlanAction:
    def __init__(self, current_day: int, action: str):
        self.current_day = current_day
        self.action = action

class TripPlanWorldModel(WorldModel):
    def __init__(self, total_days: int, base_model: LanguageModel, prompt: str, batch_size=1) -> None:
        self.total_days = total_days
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size

    def setDays(self, total_days: int):
        self.total_days = total_days
    
    def init_state(self) -> TripPlanState: # Since we can start frome any city, this should be a list.
        return TripPlanState(0, [], False)

    def step(self, state: TripPlanState, action: TripPlanAction) -> tuple[TripPlanState, dict]:
        """Take a step in the world model.
        
        :param state: the current state (see the docstring of WorldModel)
        :param action: the action to take
        :return: the next state and additional information cached for reward calculation
        """
        if state.end:
            return state, {}
        
        end_flag = False 
        new_day = action.current_day
        if new_day >= self.total_days:
            new_day = self.total_days
            end_flag = True

        new_state_history = copy.deepcopy(state.state_history)
        new_state_history.append(action.action)
        
        new_state = TripPlanState(
            current_day=new_day,
            state_history=new_state_history,
            end=end_flag
        )
        return new_state, {}
    

    def is_terminal(self, state: TripPlanState):
        return state.end
