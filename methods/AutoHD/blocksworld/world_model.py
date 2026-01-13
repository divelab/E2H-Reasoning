from typing import NamedTuple
import reasoners.benchmark.bw_utils as bwutils
import copy
from reasoners import LanguageModel, WorldModel
BWAction = str
class BWState(NamedTuple):
    """The state of the Blocksworld for ToT

    See the docstring of BlocksWorldModel for more details.
    """
    step_idx: int
    action_history: list[str]
    last_blocks_state: str
    blocks_state: str
    end: bool
    
class BlocksWorldModel(WorldModel):
    """Blocks World World Model
    State: (step_idx, action_history: [str])
    Action: e.g. "put the red block on the green block"
    """

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
        self.prompt_cache = {}

    def init_state(self) -> BWState:
        """Initialize the world model.

        :return: the initial state
        """
        return BWState(step_idx=0, action_history=[], end=False, blocks_state=bwutils.    
                       extract_init_state(self.example),last_blocks_state="") # last_blocks_state=""
    def goal_state(self,init_state) -> BWState:
        print(self.example['plan'],flush=True)
        # assert False

        extracted_actions = [action.strip() for action in  self.example['plan'].split('\n') if action.strip() and action.strip() != '[PLAN END]']
        
        goal = copy.deepcopy(init_state)
        for action in extracted_actions:
            # if action is None or action == "[PLAN END]" or action[0]==" " or action=="\n" or action[0]=="\n":
            #     continue
            print("action: ", action,flush=True)
            print(" -------------------------------------------------- ")
            print("!!!! CHECK INIT STATE",goal,flush=True)
            print(" -------------------------------------------------- ", flush=True)
            
            goal, _ = self.step(goal,action)
        # assert goal is not None, f"INVALID PLAN in self.example. Can't form goal state for heuristic calculation!"
        if goal is None:
            print("INVALID PLAN in self.example. Can't form goal state for heuristic calculation!")
        return goal
    
    def state_key(self, state:BWState):
        return state.blocks_state

    def step(self, state: BWState, action: BWAction) -> tuple[BWState, dict]:
        """Take a step in the world model.
        
        :param state: the current state (see the docstring of BlocksWorldModel)
        :param action: the action to take
        :return: the next state and additional information cached for reward calculation
        """
        state = copy.deepcopy(state)
        if state is None:
            print(f"Error state can not be none inside step!!!! Returning None. ",flush=True)
            return None, {}
        print()
        print()
        print("                     ################################################")
        print(f"                        Initial STATE {state.step_idx} - STEP inside WORLD MODEL: ", state,flush=True)
        # print("################################################",flush=True)

        print()
        # print("##############Initial Block state: ", state.blocks_state,flush=True)
        last_blocks_state = state.blocks_state
        
        if action == "[PLAN END]" or action[0]=="[" or action=="[" or action == "[ACTIONS END]":
            print("                     ############## Not applying [ action", flush=True)
            blocks_state = state.blocks_state
        else:
            print("                     ############## Applying Action: ", action,flush=True)
            blocks_state = self.update_blocks(state.blocks_state, action)

        if blocks_state is None:
            print("                     ############## New Block state: CORRUPTED!!!!!!", flush=True)
            return None, {}

        if action != "[PLAN END]" and  action != "[ACTIONS END]":
            state = BWState(step_idx=state.step_idx + 1, action_history=state.action_history + [action], 
                        blocks_state=blocks_state, last_blocks_state=last_blocks_state, end=False) #last_blocks_state=state.blocks_state,
        else:
            state = BWState(step_idx=state.step_idx + 1, action_history=state.action_history, 
                        blocks_state=blocks_state, last_blocks_state=last_blocks_state, end=True) #last_blocks_state=state.blocks_state,
        # print("################################################")
        print("                     FINAL STATE - STEP inside WORLD MODEL: ", state,flush=True)
        print("                     ################################################",flush=True)
        print()
        return state, {}
    
    def update_blocks(self, block_states: str, action: BWAction) -> str:
        """Update the block states with the action.

        :param block_states: the current block states. Note that this argument is a string,
            and it's only a part of 'BWState'
        :param action: the action to take
        :return: the updated block states
        """
        if "pick" in action:
            key = "world_update_pickup"
        elif "unstack" in action:
            key = "world_update_unstack"
        elif "put" in action:
            key = "world_update_putdown"
        elif "stack" in action:
            key = "world_update_stack"
        else:
            return None

        world_update_prompt = self.prompt[key].format(block_states, action.capitalize() + ".")
        new_state = None
        
        if world_update_prompt in self.prompt_cache:
            world_output = self.prompt_cache[world_update_prompt]
        else:
            count =0
            # Inspired and borrowed from RAP.
            while new_state is None and count<5:
                if count==0:
                    world_output = self.base_model.generate([world_update_prompt],
                                        hide_input=True, do_sample=False, max_new_tokens=130).text[0].strip()
                else:
                    world_output = self.base_model.generate([world_update_prompt],
                                        hide_input=True, do_sample=True,temperature=0.6, max_new_tokens=130).text[0].strip()
                count+=1

                try:
                    new_state = bwutils.apply_change(world_output, block_states)
                    self.prompt_cache[world_update_prompt] = world_output
                except Exception as e:
                    print(f"An error occurred in apply_change: {e}", flush=True)
                    new_state = None  # or any other fallback action
        return new_state

    def is_terminal(self, node) -> bool: 
        if node.state.end:
            return True
        if node.heuristic == 0:
            return True
        return False