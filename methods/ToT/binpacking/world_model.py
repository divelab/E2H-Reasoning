import copy
import dataclasses
import re
from typing import Optional
from reasoners import WorldModel, LanguageModel
import utils

@dataclasses.dataclass
class BinPackingState:
    item_sizes: str
    bin_sizes: str
    current: str
    history: list[str]
    output: Optional[str] = None

    def __str__(self):
        if self.output is None:
            return f'BinPackingState({repr(self.current)}, items left={repr(self.item_sizes)}, history={repr(self.history)})'
        else:
            return f'BinPackingState({repr(self.current)}, output={repr(self.output)})'


BinPackingAction = str


class BinPackingWorldModel(WorldModel):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_confidence=8,
                 batch_size=2, ) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_confidence = n_confidence

    def init_state(self) -> BinPackingState:
        print(self.example)
        new_state = BinPackingState(item_sizes=str(self.example['item_sizes']), bin_sizes=str(self.example['bin_capacity']), current=self.example['item_sizes'], history=[])
        return new_state

    def step(self, state: BinPackingState, action: BinPackingAction) -> tuple[BinPackingState, dict]:
        print('**** World Model Step ****')
        next_state = copy.deepcopy(state)
        if 'Answer' in action:
            match = re.match(r'Answer: (.*)', action)
            next_state.output = match[1] if match is not None else ''
        else:
            current, item_sizes = utils.parse_action(action)
            next_state.item_sizes = str(item_sizes)
            next_state.current = str(current)
            next_state.history.append(action)
        # print(f'Stepping {state} with {action=} to {next_state}')
        return next_state, {'next_state': next_state}

    def is_terminal(self, state: BinPackingState) -> bool:
        return state.output is not None
