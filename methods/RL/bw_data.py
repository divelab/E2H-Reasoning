from blocksworld_reward_model import BlocksWorldModel
import random 
import json
import copy

def state_visited(new_state: str, state_trace: list) -> bool:
    """
    Checks whether new_state is equivalent to any state in state_trace by using
    BlocksWorldModel.states_equal. Returns True if an equivalent state is found.
    """
    for visited in state_trace:
        equal, _ = BlocksWorldModel.states_equal(visited, new_state)
        if equal:
            return True
    return False

def generate_new_problem(problem_dict: dict, num_actions: int, starting_state_override: str = None) -> dict:
    """
    Generates a new problem instance with exactly num_actions actions.
    
    If starting_state_override is provided:
      - If it is "goal", the starting state is derived by simulating the original plan
        on the provided init state.
      - Otherwise, starting_state_override is used directly as the starting state.
    If no override is provided, a random choice is made between the "init" state and
    simulating the plan (i.e. deriving the goal) from the "init" state.
    
    The simulation avoids revisiting states (using BlocksWorldModel.states_equal) and
    only returns a problem if the generated plan has exactly num_actions actions.
    The final state is simplified by comparing it with the starting state using
    BlocksWorldModel.simplify_state_given_reference.
    
    Returns a new problem dictionary with updated "goal" and "plan".
    """
    # Determine the starting state.
    if starting_state_override is not None:
        if starting_state_override == "goal":
            # Derive starting state by simulating the plan on the init state.
            plan_str = problem_dict.get('plan', '')
            plan_lines = [line.strip() for line in plan_str.strip().split("\n")
                          if line.strip() and "[plan end]" not in line.lower()]
            current_state = problem_dict['init']
            for action in plan_lines:
                current_state = BlocksWorldModel.simulate_step(current_state, action)
            initial_state = current_state
            # Optionally print the comparison between the simulated state and expected goal.
            print(current_state)
            print(problem_dict['goal'])
            print(BlocksWorldModel.states_equal(current_state, problem_dict['goal']))
        else:
            initial_state = starting_state_override
    else:
        # Randomly choose between using the "init" state or simulating the plan ("goal")
        if random.choice(['init', 'goal']) == 'goal':
            plan_str = problem_dict.get('plan', '')
            plan_lines = [line.strip() for line in plan_str.strip().split("\n")
                          if line.strip() and "[plan end]" not in line.lower()]
            current_state = problem_dict['init']
            for action in plan_lines:
                current_state = BlocksWorldModel.simulate_step(current_state, action)
            initial_state = current_state
            print(BlocksWorldModel.states_equal(current_state, problem_dict['goal']))
        else:
            initial_state = problem_dict.get("init", "")
    
    # Now simulate num_actions from the chosen starting state.
    current_state = initial_state
    state_trace = [current_state]  # Track visited states.
    plan_actions = []

    for _ in range(num_actions):
        possible_actions = BlocksWorldModel.get_possible_actions(current_state)
        valid_actions = []
        for action in possible_actions:
            try:
                new_state = BlocksWorldModel.simulate_step(current_state, action)
                # Only consider new states that haven't been visited.
                if not any(BlocksWorldModel.states_equal(new_state, visited)[0] for visited in state_trace):
                    valid_actions.append((action, new_state))
            except Exception:
                continue
        if not valid_actions:
            # Abort if we can't generate a new state.
            break
        # Randomly choose one valid action.
        chosen_action, new_state = random.choice(valid_actions)
        plan_actions.append(chosen_action)
        current_state = new_state
        state_trace.append(new_state)

    # Only accept the problem if exactly num_actions were generated.
    if len(plan_actions) != num_actions:
        return None

    simplified_goal = BlocksWorldModel.simplify_state_given_reference(initial_state, current_state)
    new_problem = {
        "init": initial_state,
        "goal": simplified_goal,
        "plan": "\n".join(plan_actions),
        "instance_file": problem_dict.get("instance_file", ""),
        "augmentation_type": problem_dict.get("augmentation_type", ""),
        "mapping": problem_dict.get("mapping", {})
    }
    return new_problem

def _generate_unique_problems(problem_dict: dict, num_actions: int, num_required: int,
                              starting_state_override: str, generated_plans: set,
                              max_attempts: int) -> list:
    """
    Helper function to generate a list of unique problems (based on plan string)
    using an optional starting_state_override. It returns up to num_required problems,
    ensuring that the plan (as a string) is unique across the generated_problems set.
    """
    results = []
    attempts = 0
    while len(results) < num_required and attempts < max_attempts:
        new_problem = generate_new_problem(problem_dict, num_actions, starting_state_override)
        if new_problem is not None:
            plan_str = new_problem.get("plan", "").strip()
            if plan_str not in generated_plans:
                generated_plans.add(plan_str)
                results.append(new_problem)
            else:
                print('Duplicate problem generated.')
        attempts += 1
    return results

def generate_multiple_problems(problem_dict: dict, num_actions: int, num_instances: int = 1,
                               random_seed: int = None) -> list:
    """
    Generates multiple new problem instances with exactly num_actions actions each,
    ensuring that duplicate plans are not produced. The original plan from the problem_dict
    is added to the generated_plans cache to prevent duplication.
    
    Additionally, if num_actions is odd, extra problems are generated using the goal state
    as the starting state (via starting_state_override="goal") to ensure a complete distribution.
    
    Parameters:
      - problem_dict: The original problem dictionary.
      - num_actions: The required number of actions in each new problem.
      - num_instances: The number of new problems to generate (for the normal case).
      - random_seed: An optional seed for reproducibility.
    
    Returns a list of new problem dictionaries.
    """
    import random
    if random_seed is not None:
        random.seed(random_seed)
    
    original_plan = problem_dict.get("plan", "").strip()
    generated_plans = {original_plan} if original_plan else set()
    
    max_total_attempts = num_instances * 5  # Maximum attempts to generate unique problems.
    
    # Generate problems normally (without override).
    problems = _generate_unique_problems(problem_dict, num_actions, num_instances,
                                           starting_state_override=None,
                                           generated_plans=generated_plans,
                                           max_attempts=max_total_attempts)
    
    # If num_actions is odd, generate extra problems starting from the goal state.
    print(f'Generated {len(problems)} so far.')
    if num_actions % 2 == 1:
        final_problems = copy.deepcopy(problems)
        for problem in problems:
            extra = _generate_unique_problems(problem, num_actions, num_instances,
                                          starting_state_override="goal",
                                          generated_plans=generated_plans,
                                          max_attempts=max_total_attempts)
            final_problems.extend(extra)
        problems = final_problems
    print(f'Generated total of {len(problems)}')
    return problems


def generate_2step_put_down(problem_dict: dict):
    action: list = problem_dict['plan'].strip('\n').split('\n')[0] # unstack action
    plan_actions = [action]
    initial_state = problem_dict.get("init", "")
    current_state = BlocksWorldModel.simulate_step(initial_state, action)
    possible_actions = BlocksWorldModel.get_possible_actions(current_state)
    for action_i in possible_actions: # Max Len 2
        if action_i.startswith('put down'):
            new_state = BlocksWorldModel.simulate_step(current_state, action_i)
            plan_actions.append(action_i)
    
    simplified_goal = BlocksWorldModel.simplify_state_given_reference(initial_state, new_state)
    
    new_problem = {
        "init": initial_state,
        "goal": simplified_goal,
        "plan": "\n".join(plan_actions) + "\n[PLAN END]\n",
        "instance_file": problem_dict.get("instance_file", ""),
        "augmentation_type": problem_dict.get("augmentation_type", ""),
        "mapping": problem_dict.get("mapping", {})
    }
    return new_problem

def normalize_4_6step(problem_dict: dict):
    actions: list = problem_dict['plan'].replace('[PLAN END]', '').strip('\n').split('\n')
    initial_state = problem_dict.get("init", "")
    current_state = initial_state
    for action in actions:
        print(action)
        current_state = BlocksWorldModel.simulate_step(current_state, action)
    
    simplified_goal = BlocksWorldModel.simplify_state_given_reference(initial_state, current_state)
    
    new_problem = {
        "init": initial_state,
        "goal": simplified_goal,
        "plan": problem_dict['plan'],
        "instance_file": problem_dict.get("instance_file", ""),
        "augmentation_type": problem_dict.get("augmentation_type", ""),
        "mapping": problem_dict.get("mapping", {})
    }
    return new_problem
        

if __name__ == '__main__':
    with open('data/blocksworld/train_set-6-more.json') as f:
        step_2_data = json.load(f)
    
    # new_data = copy.deepcopy(step_2_data)
    print(len(step_2_data))
    new_data = []
    for problem in step_2_data:
        plan_stripped = problem['plan'].strip()
        fixed_problem = normalize_4_6step(problem)
        new_data.append(fixed_problem)
    #     # if plan_stripped.startswith('unstack the'):
    #     #     new_problem = generate_2step_put_down(problem)
    #     #     new_data.append(new_problem)
    
    print(len(step_2_data), len(new_data))
    total = new_data
    # total = []
    # num_actions =1
    # for problem in step_2_data:
    #     new_problems = generate_multiple_problems(problem, num_actions=num_actions, num_instances=1)
    #     print(new_problems)
    #     total.extend(new_problems)
    
    print(len(total))
    num_actions = 6
    with open(f'data/blocksworld/train_set-{num_actions}-all.json', 'w') as f:
        json.dump(total, f, indent=4)
            