from reasoners import LanguageModel, SearchConfig
from world_model import BWState, BWAction
from utils import make_actions
class BWConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 temperature: float = 0.8,
                 step_into_state: bool = True,
                 action_prompt: bool = True,
                 heuristic_fn: callable = None,
                 n_candidate: int = 15,
                 heuristic_search_type: str = 'test') -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.n_candidate = n_candidate
        self.temperature = temperature
        self.step_into_state = step_into_state
        self.action_prompt = action_prompt
        self.heuristic_fn = heuristic_fn
        self.search_type = heuristic_search_type
        self.prompt_cache = {}

    def get_actions(self, state: BWState) -> list[BWAction]:
        
        if self.action_prompt:
            if "the hand is empty" in state.blocks_state:
                prompts = self.prompt["next_actions_empty"].replace("<init_state>", state.blocks_state) 
            else:
                prompts = self.prompt["next_actions_holding"].replace("<init_state>", state.blocks_state) 

        else:
            actions = make_actions(state.blocks_state)
            print("UNIQUE ACTIONS:  ", actions)
            print("-------------------------------")
            return actions
            
        if prompts in self.prompt_cache:
            ouputs = self.prompt_cache[prompts]
        else:
            ouputs = self.base_model.generate([prompts],
                                              num_return_sequences=self.n_candidate,
                                              temperature=self.temperature,
                                              do_sample=True,
                                              additional_prompt="CUSTOM",
                                              hide_input=True).text
            self.prompt_cache[prompts] = ouputs

        if self.action_prompt:
            unique_actions = set()

            # Iterate through each output
            for output in ouputs:
                # Split the actions based on '\n'
                actions = output.split('\n')
                # Add each action to the set, ignoring '[ACTIONS END]'
                for action in actions:
                    action = action.strip()  
                    if action != "[ACTIONS END]" and action:
                        unique_actions.add(action)

            # Convert the set to a list to get a final set list of actions
            outputs = list(unique_actions)

        else:
            assert False

        print("UNIQUE ACTIONS Outputs: with temperature  ", self.temperature,"  :  ", outputs)
        print("-------------------------------")
        return outputs
    
    
    def heuristic(self, current_state: BWState, goal_state: BWState = None) -> float:
        
        if self.heuristic_fn is not None:
            
            try:
                my_val = self.heuristic_fn(current_state.blocks_state,goal_state.blocks_state)
            except ValueError as ve:
                print(f"ValueError occurred in heuristic function: {ve}", flush=True)
                # Handle the specific error, e.g., set my_val to a default or log the issue
                my_val = 0  # or any other fallback action
            except Exception as e:
                print(f"An unexpected error occurred in heuristic function: {e}", flush=True)
                # Handle other exceptions
                my_val = 0  # or any other fallback action
            print("MY HEURISTIC Val: ",my_val,flush=True)
            return my_val
        else:
            return 0
    
    def reward(self, state: BWState, action: BWAction, **kwargs) -> tuple[float, dict]:
        assert False
    
    
    def update_example(self, example, prompt=None) -> None:
        super().update_example(example, prompt=prompt)