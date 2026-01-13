from typing import List, Dict, Tuple, NamedTuple, Any
from reasoners import WorldModel, LanguageModel
from reasoners.algorithm import HeuristicGuidedSearchNode
from utils import *
import copy

CubeAction = str

class CubeState(NamedTuple):
    step_idx: int
    action_history: list[str]
    cube_state: str
    end: bool

class CubeWorldModel(WorldModel):
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
    
    def init_state(self) -> CubeState:
        """Initialize the world model.

        :return: the initial state
        """

        return CubeState(step_idx=0, action_history=[], end=False, cube_state=self.example)
    
    def goal_state(self, init_state:CubeState ) -> CubeState:
        return None
    
    def state_key(self, state:CubeState):
        return tuple(parseCube(state.cube_state))
    
    def step(self, state: CubeState, action: CubeAction) -> tuple[CubeState, dict]:
        state = copy.deepcopy(state)
        next_state = self.update_cubes(state.cube_state, action)
        
        if next_state is None:
            print("############## New Cube state: CORRUPTED!!!!!!", flush=True)
            return None, {}
        
        ret = CubeState(step_idx= state.step_idx + 1, action_history= state.action_history + [action],
                            cube_state=next_state, end = isSolved(parseCube(next_state)))
        
        return ret, {}
    
    def update_cubes(self, cube_state: str, action: CubeAction) -> str:
        if action not in ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"]:
            # raise ValueError("Invalid action")
            return None
        
        # ### Use GT to transfer
        return getCube(doAlgStr(parseCube(cube_state), action))
        
        
    def is_terminal(self, node: HeuristicGuidedSearchNode) -> bool: 
        if node.heuristic == 0:
            return True
        return False
    