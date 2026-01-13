from reasoners import WorldModel
from data.prontoqa.dataset import ProntoQAExample

ProntoQAState = list[str]
ProntoQAAction = str

def remove_so_prefix(s):
    if s.startswith('So '):
        return s[3:]
    return s

class ProntoQAToTWorldModel(WorldModel[ProntoQAState, ProntoQAAction, ProntoQAExample]):
    def __init__(self) -> None:
        super().__init__()
    
    def init_state(self) -> ProntoQAState:
        return []
    
    def step(self, state: ProntoQAState, action: ProntoQAAction) -> tuple[ProntoQAState, dict]:
        return state + [action], {}
    
    def is_terminal(self, state: ProntoQAState) -> bool:
        if len(state) > 0 and "The answer is" in remove_so_prefix(state[-1]):
            return True
        return False