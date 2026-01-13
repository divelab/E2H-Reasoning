import copy
import re
from typing import Literal

import numpy as np
import scipy
import torch

from reasoners import SearchConfig, LanguageModel
from world_model import BinPackingState, BinPackingAction

from prompts import output_prompt, propose_prompt, value_prompt, value_last_step_prompt, value_map
import utils

value_map = {'sure': 1, 'impossible': 0.0001}

class BinPackingConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_actions=4,
                 batch_size=2,
                 depth_limit=4,
                 temperature=0.7,
                 n_eval=5,
                 calc_reward: Literal['sampling', 'logits'] = 'sampling') -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.n_eval = n_eval
        self.value_cache = {}
        self.depth_limit = depth_limit
        self.temperature = temperature
        self.calc_reward = calc_reward

    def output_prompt_wrap(self, state: BinPackingState, example) -> str:
        return self.prompt['output_prompt'].format(item_sizes=example['item_sizes'], 
                                    bin_size=example['bin_capacity'], 
                                    history='\n'.join(state.history))

    def propose_prompt_wrap(self, state: BinPackingState, example) -> str:
        return self.prompt['propose_prompt'].format(item_sizes=state.item_sizes, 
                                     bin_size=example['bin_capacity'])

    def value_prompt_wrap(self, state: BinPackingState, example) -> str:
        return self.prompt['value_prompt'].format(bin=state.current, 
                                   capacity=example['bin_capacity'])

    def value_last_step_prompt_wrap(self, state: BinPackingState, example) -> str:
        return self.prompt['value_last_step_prompt'].format(item_sizes=example['item_sizes'], 
                                             bin_size=example['bin_capacity'], 
                                             answer=state.output)

    def retrieve_value(self, output: list[str]) -> float:
        output_names = [x.split('\n')[-1] for x in output]
        value = sum(v * output_names.count(k) for k, v in value_map.items())
        return value

    def get_actions(self, state: BinPackingState) -> list[BinPackingAction]:
        print(f'Generating actions for {state}')
        if state.item_sizes == '':
            return []
        
        if state.item_sizes == '[]': # All items have been packed.
            prompt = self.output_prompt_wrap(state, self.example)
            output = self.base_model.generate([prompt], 
                                              num_return_sequences=1, 
                                              do_sample=False, 
                                              eos_token_id='\n', 
                                              additional_prompt="CONTINUE").text[0]
            output = 'Answer: ' + output.strip()
            return [output]
        else:
            prompt = self.propose_prompt_wrap(state, self.example)
            output = self.base_model.generate([prompt], num_return_sequences=1, do_sample=False, eos_token_id='Input', additional_prompt="CONTINUE").text[0]
            output = output.strip()
            if '\n\n' in output:
                output = output.split('\n\n')[0]
            output = output.split('\n')
            # print('-----\n',output)
            actions = [x.strip() for x in output if 'left' in x]
            # set does not guarantee order, but dict does guarantee
            # we cannot use set here because torch.distributed in LLaMA requires the same order across all processes
            actions = list(dict.fromkeys(actions))
            print(actions, len(actions))
            return actions

    def _reward(self, state: BinPackingState, action: BinPackingAction) -> float:
        if state.current == '':
            return 0.
        next_state = copy.deepcopy(state)
        if 'Answer' in action:
            match = re.match(r'Answer: (.*)', action)
            next_state.output = match[1] if match is not None else ''
        else:
            current, item_sizes = utils.parse_action(action)
            next_state.item_sizes = item_sizes
            next_state.current = current

        if len(next_state.history) >= self.depth_limit:
            return 0.
        print(next_state)
        if next_state.output is None:
            prompt = self.value_prompt_wrap(next_state, self.example)
        else:
            prompt = self.value_last_step_prompt_wrap(next_state, self.example)
            print(prompt)
        if prompt in self.value_cache:
            return self.value_cache[prompt]
        if self.calc_reward == 'sampling': # This version would be run for OpenAI models mostly. 
            value_outputs = []
            for idx in range(0, self.n_eval, self.batch_size):
                n_samples = min(self.n_eval - idx, self.batch_size)
                output = self.base_model.generate([prompt], do_sample=True, temperature=self.temperature,
                                                  num_return_sequences=n_samples, additional_prompt="CONTINUE").text
                value_outputs += [o.strip().split('\n\n')[0] for o in output]
            value = self.retrieve_value(value_outputs)
        elif self.calc_reward == 'logits':
            value_keys = list(value_map.keys())
            logits = self.base_model.get_next_token_logits([prompt], value_keys)[0]
            logits = scipy.special.softmax(logits)
            value = np.sum(logits * np.array(list(value_map.values())))
        else:
            raise NotImplementedError

        self.value_cache[prompt] = value
        # print(f'Reward of {state}, {action=} is {value:.5f}')
        return value

    def fast_reward(self, state: BinPackingState, action: BinPackingAction) -> tuple[float, dict]:
        reward = self._reward(state, action)
        return reward, {'reward': reward}

    # We calculate the full reward in fast_reward in Game24SearchConfig, direct return it
    def reward(self, state: BinPackingState, action: BinPackingAction, **kwargs) -> tuple[float, dict]:
        return self.fast_reward(state, action)
