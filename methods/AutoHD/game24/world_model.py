from typing import List, Dict, Tuple, NamedTuple, Any
from reasoners import WorldModel, LanguageModel
from reasoners.algorithm import HeuristicGuidedSearchNode
import re
from collections import Counter

def self_consistency(outputs: list[str]):
    outputs = [output.strip() for output in outputs]
    output_counts = Counter(outputs)
    most_common = output_counts.most_common(1)
    if most_common:
        return most_common[0][0]
    else:
        return None

class Game24Action:
    def __init__(self, raw_action: str, new_nums: List[float]):
        self.raw_action = raw_action
        self.new_nums = new_nums
    
    def __eq__(self, other):
        if isinstance(other, Game24Action):
            return self.raw_action == other.raw_action
        return False
    
    def __hash__(self):
        return hash(self.raw_action)


class Game24State:
    def __init__(self, numbers: List[float]):
        """
        Represents the current state in the Game of 24.
        :param numbers: List of integers representing the numbers to use in calculations.
        """
        self.numbers = numbers
        self.path = ""  # Action history
        self.expression = None
        

class Game24WorldModel(WorldModel):
    def __init__(self, base_model: LanguageModel, prompt: str, batch_size=1, n_sc=1) -> None:
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_sc = n_sc
    
    # Since we can start frome any city, this should be a list. Optionally get this from LLM.
    def init_state(self) -> Game24State:
        parsed_example = [int(x) for x in self.example.split(' ')]
        initial_state = Game24State(numbers = parsed_example)
        return initial_state
    
    def output_prompt_wrap(self, state: Game24State) -> str:
        print('**World Model Output Prompt Wrap**', state.path)
        output_prompt = self.prompt['output_prompt']
        return output_prompt.format(input=self.example, history=state.path.strip())
    
    def goal_state(self, _: List[Game24State] = None) -> Game24State:
        return Game24State(numbers=[24])
    
    def step(self, state: Game24State, action: Game24Action) -> tuple[Game24State, dict]:
        """
        Apply the action to a Game24State and return the resulting state.
        :param state: The current state of the game.
        :return: A new Game24State with updated numbers.
        """
        # print('**World Model Step**', state.get_path_str(), action.new_nums)
        new_numbers = action.new_nums
        new_state = Game24State(new_numbers)
        new_state.path = state.path + '\n' + action.raw_action
        return new_state, None
    
    def is_terminal(self, node: HeuristicGuidedSearchNode) -> bool: # Figure out depth_limit
        # If heuristic evaluates to 0, and we only have one number left, we could be at a goal.
        state: Game24State = node.state
        heuristic_val = node.reward  
        if heuristic_val == 0 and len(state.numbers) == 1:
            state.expression = self.recover_expression_from_path(state)
            return True
        return False
    
    def recover_expression_from_path(self, state: Game24State) -> str:
        prompt = self.output_prompt_wrap(state)
        output = self.base_model.generate([prompt], 
                                          num_return_sequences=self.n_sc, 
                                          do_sample=False, stop='\n', 
                                          additional_prompt="CONTINUE").text
        
        # Handle Self Consistency, by default it is 1.
        output = self_consistency(outputs=output)
        return output
    
    def update_example(self, example: Any, prompt=None) -> None:
        super().update_example(example, prompt)
        
    def state_key(self, state: Game24State) -> str:
        state_key = state.numbers.copy()
        state_key.sort()
        return str(state_key)