import random
import re
from typing import List, Dict, Any, Optional
from blocksworld_reward_model import BlocksWorldModel  # assumes your BlocksWorldModel lives here

# --- Name‐augmentation helpers ---

def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def generate_mapping(original_names: List[str], candidate_list: List[str]) -> Dict[str, str]:
    if candidate_list == original_names:
        mapping = original_names.copy()
        if len(original_names) > 1:
            while mapping == original_names:
                random.shuffle(mapping)
    else:
        mapping = random.sample(candidate_list, len(original_names))
    return dict(zip(original_names, mapping))

def apply_mapping(text: str, mapping: Dict[str, str]) -> str:
    pattern = r'\b(' + '|'.join(map(re.escape, mapping.keys())) + r')\b'
    return re.sub(pattern, lambda m: mapping[m.group(1)], text, flags=re.IGNORECASE)

def adjust_number_format(text: str) -> str:
    return re.sub(r'\b(\d+)\s+block\b', r'block \1', text)

def extract_original_names(text: str) -> List[str]:
    matches = re.findall(r'\b(\w+)\s+block\b', text, flags=re.IGNORECASE)
    seen = []
    for m in matches:
        lm = m.lower()
        if lm not in seen:
            seen.append(lm)
    return seen

# --- Candidate pools for augmentation ---
COLOR_PALETTE = [
    "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "brown", "gray", "cyan", "magenta", "emerald"
]
AUG_CANDIDATES = {
    "colors": COLOR_PALETTE,
    "ordinal": None,
    "greek": ["alpha","beta","gamma","delta","epsilon","zeta","eta","theta",
              "iota","kappa","lambda","mu","nu","xi","omicron","pi",
              "rho","sigma","tau","upsilon","phi","chi","psi","omega"],
    "alphabets": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
}

# --- Single‐try problem generator (returns None on failure) ---
def try_generate_problem(num_blocks: int, num_actions: int) -> Optional[Dict[str, Any]]:
    blocks = [str(i+1) for i in range(num_blocks)]
    state, order = {}, []
    for b in blocks:
        clear = [x for x in order if x not in state.values()]
        support = random.choice(clear) if clear and random.random() < 0.5 else "table"
        state[b] = support; order.append(b)
    initial_map = state.copy()
    hand = None
    if random.random() < 1/num_blocks:
        clear = [b for b in order if b not in state.values()]
        if clear:
            hand = random.choice(clear)
            state[hand] = "hand"
    init_str = BlocksWorldModel.generate_state_string(state.copy(), hand, order)
    curr = init_str
    seen = {curr}
    plan_actions = []
    moved = set()

    for _ in range(num_actions):
        poss = BlocksWorldModel.get_possible_actions(curr)
        valid = []
        for a in poss:
            try:
                nxt = BlocksWorldModel.simulate_step(curr, a)
                if any(BlocksWorldModel.states_equal(nxt, s)[0] for s in seen):
                    continue
                valid.append((a, nxt))
            except:
                pass
        if not valid:
            return None
        by_type = {"unstack":[],"pickup":[],"stack":[],"putdown":[]}
        for a,nxt in valid:
            t = a.split()[0]
            key = {"unstack":"unstack","pick":"pickup","stack":"stack","put":"putdown"}[t]
            by_type[key].append((a,nxt))
        types = [t for t,lst in by_type.items() if lst]
        a,nxt = random.choice(by_type[random.choice(types)])
        prev_map,_,_ = BlocksWorldModel.parse_initial_state(curr)
        next_map,_,_ = BlocksWorldModel.parse_initial_state(nxt)
        moved.update(b for b in blocks if prev_map[b] != next_map[b])
        plan_actions.append(a)
        curr = nxt
        seen.add(curr)

    if len(plan_actions) != num_actions:
        return None
    goal_str = BlocksWorldModel.simplify_state_given_reference(init_str, curr)
    final_map,_,_ = BlocksWorldModel.parse_initial_state(curr)
    if any(final_map[b] == initial_map[b] for b in moved):
        return None

    # Name augmentation
    combined = f"{init_str}\n" + "\n".join(plan_actions) + f"\n{goal_str}"
    names = extract_original_names(combined)
    aug_type = random.choice(list(AUG_CANDIDATES.keys()))
    if aug_type == "ordinal":
        candidates = [ordinal(i+1) for i in range(len(names))]
    elif aug_type == "shuffled":
        candidates = names.copy()
    else:
        candidates = AUG_CANDIDATES[aug_type]
    mapping = generate_mapping(names, candidates)
    return {
        "init": adjust_number_format(apply_mapping(init_str, mapping)),
        "plan": adjust_number_format(apply_mapping("\n".join(plan_actions), mapping)),
        "goal": adjust_number_format(apply_mapping(goal_str, mapping)),
        "mapping": mapping,
        "augmentation_type": aug_type
    }

# --- Detector for redundancy ---
def detect_redundant_problems(dataset: List[Dict[str, Any]], new_probs = None) -> List[int]:
    bad = []
    for idx, prob in enumerate(dataset):
        init_map, _, _ = BlocksWorldModel.parse_initial_state(prob["init"])
        curr = prob["init"]
        moved = set()
        for a in prob["plan"].splitlines():
            if not a.strip(): continue
            nxt = BlocksWorldModel.simulate_step(curr, a)
            pm,_,_ = BlocksWorldModel.parse_initial_state(curr)
            nm,_,_ = BlocksWorldModel.parse_initial_state(nxt)
            moved.update(b for b in init_map if pm[b] != nm[b])
            curr = nxt
        fm,_,_ = BlocksWorldModel.parse_initial_state(curr)
        if any(fm[b] == init_map[b] for b in moved):
            bad.append(idx)
    return bad

# --- Batch generator with prints and retry ---
def generate_random_problem_batch(num_instances: int,
                                  num_actions: int,
                                  min_blocks: int = 2,
                                  max_blocks: Optional[int] = None,
                                  max_attempts: int = 10000
                                 ) -> List[Dict[str, Any]]:
    if max_blocks is None:
        max_blocks = len(AUG_CANDIDATES["colors"])
    batch = []
    attempts = 0
    while len(batch) < num_instances and attempts < max_attempts:
        attempts += 1
        n = random.randint(min_blocks, max_blocks)
        prob = try_generate_problem(n, num_actions)
        if prob is None:
            # print(f"Attempt {attempts}: failed to generate a valid {num_actions}-step problem; retrying...")
            continue
        batch.append(prob)
        print(f"Created problem {len(batch)}/{num_instances} after {attempts} attempts")
    if len(batch) < num_instances:
        print(f"Warning: only created {len(batch)}/{num_instances} problems after {attempts} attempts")
    return batch
from collections import deque
def find_optimal_plan(init_str: str, goal_str: str) -> Optional[List[str]]:
    """
    Run a BFS from init_str to goal_str, returning the first (i.e. shortest)
    sequence of actions that reaches the goal, or None if no solution.
    """
    seen = {init_str}
    queue = deque([(init_str, [])])             # pairs of (state, actions so far)

    while queue:
        state, actions = queue.popleft()
        # goal‐test
        if BlocksWorldModel.states_equal(state, goal_str)[0]:
            return actions

        # expand
        for a in BlocksWorldModel.get_possible_actions(state):
            try:
                nxt = BlocksWorldModel.simulate_step(state, a)
            except Exception:
                continue
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, actions + [a]))
    return None


def is_plan_optimal_length(problem: dict) -> bool:
    """
    Given a problem dict with keys 'init', 'plan', 'goal', returns True
    if the length of the provided plan exactly matches the optimal length.
    """
    # parse the user‐provided plan
    user_plan = [line for line in problem["plan"].splitlines() if line.strip()]
    # compute the optimal plan
    optimal = find_optimal_plan(problem["init"], problem["goal"])
    if optimal is None:
        raise RuntimeError("No solution found from init to goal!")
    # compare lengths
    return len(user_plan) == len(optimal)

def check_all_plans(batch):
    indices = []
    import time
    start = time.time()
    for idx, prob in enumerate(batch):
        if not is_plan_optimal_length(prob):
            print(f'Bad Problem -  {idx} -  total {len(indices)+1}')
            indices.append(idx)
    print(f'Time: {time.time() - start}')
    return indices

if __name__ == "__main__":
    import json
    NUM_INSTANCES = 50
    # with open('train_set-8-complete-correct.json', 'r') as f:
    #     data = json.load(f)
    # print(len(detect_redundant_problems(data)))
    for na in [1]:
        # print(f"\nGenerating {NUM_INSTANCES} problems with plan length {na}")
        batch = generate_random_problem_batch(NUM_INSTANCES, na, min_blocks=2, max_blocks=6)
        # with open(f'train_set-{na}-complete-correct.json', 'r') as f:
        #     data = json.load(f)
        bad_indices = detect_redundant_problems(batch)
        # bad_indices = check_all_plans(data)
        # batch = data
        while bad_indices:
            print(f"Detected {len(bad_indices)} redundant problem(s); regenerating those...")
            # regenerate exactly as many problems as needed
            new_probs = generate_random_problem_batch(len(bad_indices), na, min_blocks=2, max_blocks=6)
            # replace bad ones
            for idx, new_prob in zip(bad_indices, new_probs):
                batch[idx] = new_prob
            # recheck
            bad_indices = detect_redundant_problems(batch)
            non_optimal_indices = check_all_plans(data)
            if bad_indices and non_optimal_indices:
                bad_indices = list(set(bad_indices).union(set(non_optimal_indices)))
                print(f'combined length - {len(bad_indices)}')
            elif non_optimal_indices:
                bad_indices = non_optimal_indices
        filename = f"train_set-{na}-test.json"
        with open(filename, "w") as f:
            json.dump(batch, f, indent=2)
        print(f"Wrote {len(batch)} problems to {filename}")
