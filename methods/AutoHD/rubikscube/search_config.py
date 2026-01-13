from reasoners import LanguageModel, SearchConfig
from typing import List, Callable
import numpy as np
from utils import parseCube
from world_model import CubeState, CubeAction, CubeWorldModel


class CubeConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 temperature: float = 0.8,
                 n_candidate: int = 5,
                 heuristic_fn: Callable = None,) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.n_candidate = n_candidate
        self.temperature = temperature
        self.heuristic_fn = heuristic_fn
    
    def get_actions(self, state: CubeState) -> list[CubeAction]:
        prompts = self.prompt['next_actions'].replace("<cube_state>", state.cube_state) 
        
        
        outputs = self.base_model.generate([prompts],
                                          num_return_sequences=self.n_candidate,
                                          temperature=self.temperature,
                                          do_sample=True,
                                          hide_input=True,
                                          eos_token_id = '\n\n',
                                          top_p = 0.95,
                                          system_prompt = "Please continue to provide all possible moves for the current state start with [All Possible Moves], following the format of previous examples. Don't say any other words.\n\n").text
        
        print("ACTIONS Outputs: ", outputs)
        # print("-------------------------------")
        
        unique_actions = set()
        for output in outputs:
            prefix = "[All Possible Moves]"
            if prefix in output:
                output = output.split(prefix, 1)[1].strip()
            actions = output.split(",")
            for action in actions:
                action = action.strip()
                unique_actions.add(action)
        
        outputs = list(unique_actions)
        print("Unique ACTIONS Outputs: ", outputs)
        return outputs

    def heuristic(self, current_state:CubeState, goal_state) -> float:     
        if self.heuristic_fn is not None:
            state = parseCube(current_state.cube_state)
            try:
                return self.heuristic_fn(state)  
            except:
                return 0
            
        
        State = parseCube(current_state.cube_state)
        
        # llama 70 B
        heuristic_val = 0
        face_indices = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]

        misaligned_faces = 0
        max_misaligned_squares = 0

        for start, end in face_indices:
            face = State[start:end]
            most_common_color = np.bincount(face).argmax()
            misaligned_squares = np.sum(face != most_common_color)
            
            if misaligned_squares > 0:
                misaligned_faces += 1
                max_misaligned_squares = max(max_misaligned_squares, misaligned_squares)

        heuristic_val = misaligned_faces + max_misaligned_squares
    
        return heuristic_val        
              
    def reward(self, state:CubeState, action:CubeAction, **kwargs) -> tuple[float, dict]:
        assert False