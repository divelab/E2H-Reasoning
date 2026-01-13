from typing import Generic
from collections import defaultdict
from .. import SearchAlgorithm, WorldModel, SearchConfig, State, Action, Example
from typing import NamedTuple, List, Tuple, Callable, Any, Union, Optional
import numpy as np
import warnings
import random
from copy import deepcopy
import itertools
import heapq

# from heuristics import bw_heuristic

class HeuristicGuidedSearchNode:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self,
                 state: State,
                 action: Action,
                 reward: float,
                 parent: Optional['HeuristicGuidedSearchNode'] = None,
                 children: Optional[List['HeuristicGuidedSearchNode']] = None,
                 heuristic = None
                 ) -> None:
        self.id = next(HeuristicGuidedSearchNode.id_iter)
        self.state = state
        self.action = action
        self.reward = reward
        self.parent = parent
        self.children = children if children is not None else []
        self.heuristic = heuristic

    def add_child(self, child: 'HeuristicGuidedSearchNode'):
        self.children.append(child)

    def get_trace(self) -> List[Tuple[Action, State, float]]:
        """ Returns the sequence of actions and states from the root to the current node """
        node, path = self, []
        while node is not None:
            path.append((node.action, node.state, node.reward))
            node = node.parent
        # Reverse the path to get actions and states in order
        path = path[::-1]
        return path
    
    def __lt__(self, other):
        return self.reward < other.reward


class HeuristicGuidedSearchResult(NamedTuple):
    terminal_node: HeuristicGuidedSearchNode
    terminal_state: State
    cum_reward: float
    tree: HeuristicGuidedSearchNode
    trace: List[Tuple[Action, State, float]]


class HeuristicGuidedSearch(SearchAlgorithm, Generic[State, Action]):
    def __init__(self, max_depth: int, sampling_strategy: str = 'argmax',
                 replace: Optional[bool] = None, temperature: Optional[float] = None,
                 temperature_decay: Optional[float] = None, reject_sample: Optional[bool] = None,
                 reject_min_reward: Optional[float] = None, unbiased: Optional[bool] = None,
                 reward_aggregator: Union[Callable[[List[Any]], float], str] = 'last', action_dedup: bool = False,
                 early_terminate: bool = True, return_beam: bool = False, call_old = False, n_iters=1, terminal_beam_size=1, add_cost:bool = False, **kwargs) -> None:
        # Initialize the HeuristicGuidedSearch class
        super().__init__(**kwargs)
        self.max_depth = max_depth
        self.sampling_strategy = sampling_strategy
        # print(f'sampling_strategy: {sampling_strategy}', flush=True)
        
        self.replace = replace
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.reject_sample = reject_sample
        self.reject_min_reward = reject_min_reward
        self.unbiased = unbiased
        self.reward_aggregator = reward_aggregator
        self.action_dedup = action_dedup
        self.early_terminate = early_terminate
        self.return_beam = return_beam
        self.call_old = call_old # Temporary flag to call old version of beam search
        self.num_iters = n_iters
        self.terminal_beam_size = terminal_beam_size
        # Initializing the reward_aggregator based on the provided argument
        self._initialize_reward_aggregator()

        # Postprocessing after initialization
        self._post_initialization()
        self.add_cost = add_cost

    def _initialize_reward_aggregator(self):
        # how to aggregate the reward list
        if self.reward_aggregator == 'cumulative' or self.reward_aggregator == 'accumulative':
            self.reward_aggregator = lambda x: sum(x)
        elif self.reward_aggregator == 'mean' or self.reward_aggregator == 'average':
            self.reward_aggregator = lambda x: sum(x) / len(x)
        elif isinstance(self.reward_aggregator, str) and self.reward_aggregator.startswith('last'):
            self.reward_aggregator = lambda x: x[-1]
        else:
            # if the reward_aggregator is a string but not the above, raise error
            if isinstance(self.reward_aggregator, str):
                raise NotImplementedError(f"Reward aggregator {self.reward_aggregator} is not implemented.")

    def _post_initialization(self):
        # if the temperature is set to 0, then we force the sampling strategy to be argmax
        if self.temperature and self.temperature < 1e-4:
            self.sampling_strategy = 'argmax'
            warnings.warn(f"Temperature is set to 0, sampling strategy is forced to be argmax.")

        # argmax = greedy = deterministic = topk
        if self.sampling_strategy in ['greedy', 'deterministic', 'topk']:
            self.sampling_strategy = 'argmax'

        # if sampling strategy not in argmax or stochastic, just use argmax
        if self.sampling_strategy not in ['argmax', 'stochastic']:
            self.sampling_strategy = 'argmax'
            warnings.warn(f"Sampling strategy only supports argmax or stochastic, but got {self.sampling_strategy}. \
                            Sampling strategy is changed to argmax automatically.")

        # if early_terminate is set to False, we need to inform the user that we will return the beam instead of the best trace
        if not self.early_terminate:
            self.return_beam = True
            warnings.warn(
                f"early_terminate is set to False, BeamSearch will return the beam instead of the best trace.")


    def __call__(self, world: WorldModel[State, Action, State], config: SearchConfig[State, Action, State]):
        if self.call_old:
            return self.__call__old(world, config)
        return self.__call__dfs(world, config)

    def __call__dfs(self, world: WorldModel[State, Action, State], config: SearchConfig[State, Action, State]):
        HeuristicGuidedSearchNode.reset_id()
        init_state = world.init_state()
        goal_state = world.goal_state(init_state)
        terminal_beam = []
        a_star_heap = []
        closed_set = set()
        best_result = None
        if isinstance(init_state, list):
            for i, state in enumerate(init_state):
                node = HeuristicGuidedSearchNode(state=state, action=None, reward=config.heuristic(state, goal_state))
                if i==0:
                    root_node = HeuristicGuidedSearchNode(state=state, action=None, reward=config.heuristic(state, goal_state))
                    
                # Initialize current beam with initial state
                heapq.heappush(a_star_heap, (node.reward,[], node))
        else:
            root_node = HeuristicGuidedSearchNode(state=init_state, action=None, reward=config.heuristic(init_state, goal_state))
            # Initialize current beam with initial state
            best_result = HeuristicGuidedSearchResult(
                    terminal_node=root_node,
                    terminal_state=root_node.state,
                    cum_reward=root_node.reward,
                    trace=root_node.get_trace(),
                    tree=None
            )
            heapq.heappush(a_star_heap, (root_node.reward, [], root_node))
        iters = self.max_depth * self.num_iters # State Space Search
        print(f'Running HeuristicGuidedSearch for {iters} search iterations.')
        for i in range(iters):
            if not a_star_heap:
                break
            # print(f'Iteration: {i}')
            # print(a_star_heap)
            current_node = heapq.heappop(a_star_heap)
        
            _, reward_list, search_node = current_node
            
            state = search_node.state
            
            state_key = world.state_key(state)
            # if state_key in closed_set:
            #     continue
            closed_set.add(state_key)
            
            # Deduplication for states should be done at config level.
            actions = config.get_actions(state)
            # print(f'Number of Actions generated: {len(actions)}')
            for j, action in enumerate(actions):
                # print(f'Executing Action {j}')
                next_state, aux = world.step(state, action)
                if next_state is None:
                    continue
                
                heuristic = config.heuristic(next_state, goal_state)
                if self.add_cost:
                    child_node = HeuristicGuidedSearchNode(state=next_state, action=action, reward=heuristic + state.step_idx + 1, heuristic=heuristic)
                else:
                    child_node = HeuristicGuidedSearchNode(state=next_state, action=action, reward=heuristic, heuristic=heuristic)
                search_node.add_child(child_node)
                # If the heauristic evaluates to zero, or the max search has exhausted.
                if self.early_terminate and world.is_terminal(child_node): 
                    print('Solution Found Adding to Terminal Beam')
                    terminal_beam.append(HeuristicGuidedSearchResult(
                        terminal_node=child_node,
                        terminal_state=child_node.state,
                        cum_reward=child_node.reward,
                        trace=child_node.get_trace(),
                        tree=root_node
                    ))
                    if len(terminal_beam) == self.terminal_beam_size:
                        print('Terminal Beam Size Reached, returning candidate solutions')
                        return terminal_beam 
                if self.add_cost:
                    heapq.heappush(a_star_heap, (heuristic + state.step_idx + 1, reward_list, child_node))
                else:
                    heapq.heappush(a_star_heap, (heuristic, reward_list, child_node))
                # print(a_star_heap)
        
        if len(terminal_beam) == 0:
            print('NO Solution Found')
            return [best_result]
        
        return terminal_beam