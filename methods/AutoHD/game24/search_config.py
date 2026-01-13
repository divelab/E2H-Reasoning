from reasoners import WorldModel, LanguageModel, SearchConfig
from typing import List, Callable
from reasoners.lm.openai_model import OpenAIModel
import re
import utils
from world_model import Game24WorldModel, Game24State, Game24Action


class Game24Config(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 temperature: float = 0.8,
                 n_candidate: int = 4, # we probably don't need this, but still need to verify
                 heuristic_fn: Callable[[Game24State], float] = None,
                 heuristic_search_type: str = 'test') -> None:
        """
        Configuration for searching a solution to the Game of 24.
        :param base_model: The language model for generating prompts and evaluating actions.
        :param prompt: A dictionary containing prompt templates.
        :param temperature: Controls randomness in the model's output.
        :param n_candidate: Number of candidate actions to generate.
        :param heuristic_fn: Optional heuristic function to guide towards the solution.
        """
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt
        self.temperature = temperature
        self.n_candidate = n_candidate
        self.heuristic_fn = heuristic_fn
        self.search_type = heuristic_search_type
        self.action_cache = {}


    def generate_action_prompt(self, state: Game24State) -> str:
        """
        Creates a prompt for the LLM to generate all possible actions for the given state.
        state: Current game state with available numbers.
        Sample returned string format for actions: "Action: +(8, 3)"
        """
        propose_prompt_template = self.prompt['action_prompt']
        input_numbers = ' '.join(str(num) for num in state.numbers)
        prompt = propose_prompt_template.format(input=input_numbers)
        return prompt

    def get_actions(self, state: Game24State) -> List[Game24Action]:
        """
        Uses the LLM to generate all possible actions for the current state.
        :param state: Current game state.
        :return: List of Game24Action objects.
        """
        if len(state.numbers) ==1: # Final number.
            return []
        prompt = self.generate_action_prompt(state)
        # Cache the prompt for Heuristic Search, for fair evaluation and fast lookup.
        if prompt in self.action_cache:
            print(f"Cache Hit for: {state.numbers}")
            return self.action_cache[prompt]
        
        # Todo add multiple iteration support (self consistency)
        if self.search_type == 'val':
            response = utils.generate_actions(state.numbers)
        else:
            response = self.base_model.generate([prompt],
                temperature=self.temperature,
                num_return_sequences=1,
                additional_prompt="CONTINUE",
                do_sample=False,
            ).text
            print(response)
            
        # Process each extracted text
        all_raw_actions = []
        for text in response:
            raw_actions_list = self.simple_parse_LLM_output(text)
            all_raw_actions.extend(raw_actions_list)
        
        deduplicated_actions = set(all_raw_actions)
        # deduplicated_actions = self.deduplicate_actions(state, all_raw_actions)
        if prompt not in self.action_cache: # Cache the prompt for Heuristic Search, for fair evaluation and fast lookup.
            self.action_cache[prompt] = deduplicated_actions    
        print(f"Final List has {len(deduplicated_actions)} Actions.\n")
        return deduplicated_actions

    def simple_parse_LLM_output(self, response: str): # Need to simplify this parse function, and modify Game24Action
        actions = []
        for line in response.splitlines():
            if '=' not in line:
                # print('No equal sign in line', line)
                continue
            try:
                _, new_nums = line.split('=')
                match = re.match(r'.*\(left: (.*)\)', new_nums)
                new_num_list = match[1] if match is not None else ''
                new_num_list = [float(x) for x in new_num_list.split(' ')]
                raw_action = line.strip()
            except Exception as e:
                print(f'Error parsing action: {e}')
                continue
            actions.append(Game24Action(raw_action, new_num_list))
        return actions

    def heuristic(self, current_state: Game24State, goal_state: Game24State = None) -> float:
        # Use the LLM generated heuristic.
        h_val = 0
        if self.heuristic_fn is not None:
            try:
                h_val = self.heuristic_fn(current_state.numbers)
            except:
                h_val = 0
        return h_val              
    
    def reward(self, state: Game24State, action: Game24Action, **kwargs) -> tuple[float, dict]:
        assert False
