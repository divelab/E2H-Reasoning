import re
import json

class BlocksWorldModel:
    def __init__(self, init_state: str, goal: str, plan: str):
        self.init_state = init_state
        self.goal = goal
        self.plan = plan

    # --- Normalization & Denormalization ---
    @staticmethod
    def normalize_input_string(s: str) -> str:
        """
        Replace occurrences of 'block <number>' with just '<number>'.
        For example: "the block 2 is clear" becomes "the 2 is clear".
        Colored blocks (like "gold block") remain unchanged.
        """
        return re.sub(r'\bblock (\d+)\b', r'\1 block', s, flags=re.IGNORECASE)

    @staticmethod
    def normalize_block_name(name: str) -> str:
        """
        Normalize a block name by stripping whitespace, converting to lowercase,
        and removing a trailing " block" if present.
        For example, "C block" or "c block" becomes "c".
        """
        name = name.strip().lower()
        if name.endswith(" block"):
            name = name[:-6].strip()
        return name

    @staticmethod
    def denormalize_block_name(name: str) -> str:
        """
        Convert an internal block name back to the proper output format.
        If the name is numeric, return "block <num>"; otherwise, return "<name> block".
        (Note: Colored names will appear in lowercase.)
        """
        return f"{name} block"

    # --- Parsing Functions ---
    @classmethod
    def parse_initial_state(cls, state_str: str):
        """
        Parses the initial state string into:
          - state: a dictionary mapping each normalized block to its location
                   ("table", another block, or "hand")
          - hand: None if empty, or the block currently held
          - blocks_order: a list of block names (normalized) in order of appearance.
        """
        state_str = cls.normalize_input_string(state_str)
        normalized = state_str.replace(" and ", ", ")
        facts = [fact.strip() for fact in normalized.split(",")]
        
        state = {}  # maps block -> location ("table", another block, or "hand")
        hand = None
        blocks_order = []

        def add_block(b: str) -> str:
            bn = cls.normalize_block_name(b)
            if bn not in blocks_order:
                blocks_order.append(bn)
            return bn

        for fact in facts:
            # Hand facts.
            if re.fullmatch(r"the hand is empty", fact, re.IGNORECASE):
                hand = None
            elif m := re.match(r"the hand is holding the (.+?)(?: block)?$", fact, re.IGNORECASE):
                block = cls.normalize_block_name(m.group(1))
                hand = block
                state[block] = "hand"
                add_block(block)
            # Clear facts.
            elif m := re.match(r"the (.+?)(?: block)? is clear$", fact, re.IGNORECASE):
                block = cls.normalize_block_name(m.group(1))
                state.setdefault(block, None)
                add_block(block)
            # "On top of" relations.
            elif m := re.match(r"the (.+?)(?: block)? is on top of the (.+?)(?: block)?$", fact, re.IGNORECASE):
                block = cls.normalize_block_name(m.group(1))
                support = cls.normalize_block_name(m.group(2))
                state[block] = support
                add_block(block)
                add_block(support)
            # "On the table" relations.
            elif m := re.match(r"the (.+?)(?: block)? is on the table$", fact, re.IGNORECASE):
                block = cls.normalize_block_name(m.group(1))
                state[block] = "table"
                add_block(block)
            # Unrecognized facts are ignored.
        return state, hand, blocks_order

    @classmethod
    def parse_action(cls, action_str: str):
        """
        Parses an action string into a tuple describing the action.
        Supported actions:
          - "unstack the <block> from on top of the <support>"
          - "pick up the <block>"
          - "stack the <block> on top of the <support>"
          - "put down the <block>"
        The action string is normalized before parsing.
        """
        action_str = cls.normalize_input_string(action_str)
        action_str = action_str.lower().strip()
        # print(action_str)
        patterns = [
            (
                r"^unstack the (\w+) block from on top of the (\w+) block$",
                lambda m: ("unstack", cls.normalize_block_name(m.group(1)), cls.normalize_block_name(m.group(2)))
            ),
            (
                r"^pick up the (\w+) block$",
                lambda m: ("pickup", cls.normalize_block_name(m.group(1)))
            ),
            (
                r"^stack the (\w+) block on top of the (\w+) block$",
                lambda m: ("stack", cls.normalize_block_name(m.group(1)), cls.normalize_block_name(m.group(2)))
            ),
            (
                r"^put down the (\w+) block$",
                lambda m: ("putdown", cls.normalize_block_name(m.group(1)))
            ),
        ]
        for pattern, action_fn in patterns:
            m = re.match(pattern, action_str)
            if m:
                return action_fn(m)
        raise ValueError("Action not recognized or unsupported.")

    @classmethod
    def parse_goal(cls, goal_str: str):
        """
        Parses the goal string into a dictionary mapping each normalized block
        to its expected location.
        Supports "is on top of" and "is on the table" relations.
        """
        goal_str = cls.normalize_input_string(goal_str)
        goal_str = goal_str.replace(" and ", ", ")
        facts = [fact.strip() for fact in goal_str.split(",")]
        goal_dict = {}
        for fact in facts:
            m = re.match(r"the (.+?)(?: block)? is on top of the (.+?)(?: block)?$", fact, re.IGNORECASE)
            if m:
                block = cls.normalize_block_name(m.group(1))
                support = cls.normalize_block_name(m.group(2))
                goal_dict[block] = support
            else:
                m = re.match(r"the (.+?)(?: block)? is on the table$", fact, re.IGNORECASE)
                if m:
                    block = cls.normalize_block_name(m.group(1))
                    goal_dict[block] = "table"
        return goal_dict

    # --- Simulation Functions ---
    @staticmethod
    def is_clear(block: str, state: dict, hand: str) -> bool:
        """
        A block is clear if no other block is resting on top of it.
        (A block held by the hand is not considered clear.)
        """
        if state.get(block) == "hand":
            return False
        for b, loc in state.items():
            if loc == block:
                return False
        return True

    @classmethod
    def simulate_action(cls, state: dict, hand: str, blocks_order: list, action_tuple: tuple):
        """
        Applies the given action to the state.
        Supported actions: "unstack", "pickup", "stack", "putdown".
        Returns the updated state and hand.
        """
        action = action_tuple[0]
        if action == "unstack":
            block, support = action_tuple[1], action_tuple[2]
            if state.get(block) != support:
                raise Exception(f"Precondition failed: {cls.denormalize_block_name(block)} is not on {cls.denormalize_block_name(support)}.")
            if not cls.is_clear(block, state, hand):
                raise Exception(f"Precondition failed: {cls.denormalize_block_name(block)} is not clear.")
            if hand is not None:
                raise Exception("Precondition failed: hand is not empty.")
            state[block] = "hand"
            hand = block
        elif action == "pickup":
            block = action_tuple[1]
            if state.get(block) != "table":
                raise Exception(f"Precondition failed: {cls.denormalize_block_name(block)} is not on the table.")
            if not cls.is_clear(block, state, hand):
                raise Exception(f"Precondition failed: {cls.denormalize_block_name(block)} is not clear.")
            if hand is not None:
                raise Exception("Precondition failed: hand is not empty.")
            state[block] = "hand"
            hand = block
        elif action == "stack":
            block, support = action_tuple[1], action_tuple[2]
            if hand != block:
                raise Exception(f"Precondition failed: hand is not holding {cls.denormalize_block_name(block)}.")
            if not cls.is_clear(support, state, hand):
                raise Exception(f"Precondition failed: {cls.denormalize_block_name(support)} is not clear.")
            state[block] = support
            hand = None
        elif action == "putdown":
            block = action_tuple[1]
            if hand != block:
                raise Exception(f"Precondition failed: hand is not holding {cls.denormalize_block_name(block)}.")
            state[block] = "table"
            hand = None
        else:
            raise Exception("Action not supported.")
        return state, hand

    @classmethod
    def generate_state_string(cls, state: dict, hand: str, blocks_order: list) -> str:
        """
        Generates a state description string in the same format as the input.
        The output includes clear facts, the hand status, and location facts.
        """
        facts = []
        for block in blocks_order:
            if state.get(block) != "hand" and cls.is_clear(block, state, hand):
                facts.append(f"the {cls.denormalize_block_name(block)} is clear")
        if hand is None:
            facts.append("the hand is empty")
        else:
            facts.append(f"the hand is holding the {cls.denormalize_block_name(hand)}")
        for block in blocks_order:
            loc = state.get(block)
            if loc is None or loc == "hand":
                continue
            elif loc == "table":
                facts.append(f"the {cls.denormalize_block_name(block)} is on the table")
            else:
                facts.append(f"the {cls.denormalize_block_name(block)} is on top of the {cls.denormalize_block_name(loc)}")
        if not facts:
            return ""
        if len(facts) == 1:
            return facts[0]
        return ", ".join(facts[:-1]) + " and " + facts[-1]

    # --- Single-Step Simulation ---
    @classmethod
    def simulate_step(cls, state_str: str, action_str: str) -> str:
        """
        Simulates a single action (step) given an initial state string and an action string.
        Returns the new state as a description string.
        """
        state, hand, blocks_order = cls.parse_initial_state(state_str)
        action_tuple = cls.parse_action(action_str)
        new_state, new_hand = cls.simulate_action(state, hand, blocks_order, action_tuple)
        return cls.generate_state_string(new_state, new_hand, blocks_order)

    # --- Plan Simulation & Goal Testing ---
    def simulate_plan(self, plan=None, simplify=False):
        """
        Simulates the sequence of actions in the plan starting from the initial state.
        Returns the final internal state, hand, blocks_order, and a description string.
        """
        if plan is None:
            plan = self.plan
        lines = [line.strip() for line in plan.strip().split("\n")
                 if line.strip() and "[plan end]" not in line.lower()]
        state = self.init_state
        for action_line in lines:
            # action_tuple = self.parse_action(action_line)
            state = self.simulate_step(state, action_line)
        final_state_str = state
        if simplify:
            final_state_str = self.simplify_state_given_reference(self.init_state, final_state_str)
        return final_state_str

    def check_goal(self, state: str):
        """
        Checks if the provided state meets the goal conditions.
        Returns a tuple: (goal_reached (bool), missing_conditions (list of strings)).
        """
        return self.states_equal(state, self.goal)

    def test(self):
        """
        Runs the plan starting from the initial state, then checks if the final state meets the goal.
        Returns a tuple: (final_state_str, goal_reached, missing_conditions).
        """
        state = self.simulate_plan()
        goal_reached, missing_conditions = self.check_goal(state)
        return state, goal_reached, missing_conditions

    # --- Reward-based Simulation for RL ---
    def get_number_of_steps(self, plan):
        return sum(1 for line in plan.strip().splitlines() if line.strip() and "[plan end]" not in line.lower())
    
    def simulate_plan_with_reward_1(self, true_plan: str):
        """
        Simulates the plan step by step. If any action is physically unachievable,
        returns a reward of 0. If all actions are valid but the goal is not reached,
        returns a reward of 0.1. If the final state meets the goal, returns a reward of 1.
        """
        print('---- Simulating Plan Rewards ----')
        if len(self.plan.strip().split("\n")) == 0:
            print('Empty plan.')
            return 0.0
        num_steps_extracted_plan = self.get_number_of_steps(self.plan)
        num_steps_true_plan = self.get_number_of_steps(true_plan)
            
            # Since number of steps cannot be less than the true plan, the BW reward is 0.0.
        print(f"Number of steps extracted: {num_steps_extracted_plan}, true: {num_steps_true_plan}")
        if num_steps_extracted_plan < num_steps_true_plan:
            print(f"ERROR ---Number of steps mismatch!----")
            return 0.0
        
        try:
            lines = [line.replace('<', '').replace('>', '').strip() for line in self.plan.strip().split("\n")
                     if line.strip() and "[plan end]" not in line.lower()]
            print(lines)
            curr_state = self.init_state
            for action_line in lines:
                curr_state = self.simulate_step(curr_state, action_line)
        except ValueError as ve:
            print(f'Invalid Action - {ve}')
            return 0.0
        except Exception as e:
            print(f'Physically Impossible Action - {e}')
            return 0.1
        # If actions were valid, check the goal.
        goal_reached, _ = self.check_goal(curr_state)
        if goal_reached and num_steps_extracted_plan == num_steps_true_plan:
            print('Goal Reached! Optimal Plan!')
            return 2.0 # Goal reached with the right number of steps
        elif goal_reached and num_steps_extracted_plan > num_steps_true_plan:
            print(f'Goal Reached! Suboptimal Plan! True Plan: {num_steps_true_plan}, Extracted Plan: {num_steps_extracted_plan}')
            return 1.0 # Goal reached with more steps
        elif not goal_reached and num_steps_extracted_plan == num_steps_true_plan:
            print(f'Goal Not Reached. Correct Number of Steps. True Plan: {num_steps_true_plan}, Extracted Plan: {num_steps_extracted_plan}')
            return 0.5 # Goal not reached with the right number of steps
        elif not goal_reached and num_steps_extracted_plan > num_steps_true_plan:
            print(f'Goal Not Reached. Incorrect Number of Steps. True Plan: {num_steps_true_plan}, Extracted Plan: {num_steps_extracted_plan}')
            return 0.3 # Goal not reached with more steps
        
    def simulate_plan_with_reward(self, true_plan: str) -> float:
        """
        Simulates the plan step by step, awarding 1 point for each valid action.
        After each valid action, it computes an Intersection over Union (IoU) score between 
        the current state and the goal, using the same parse_initial_state function. 
        The IoU is based on the set of (block, location) conditions, which naturally
        accounts for the hand if a block is being held.

        If an action is physically impossible, the simulation stops immediately 
        and returns the reward accumulated so far plus the IoU of the last valid state.
        If all actions are valid, the final reward is:
            (number of valid actions) + (IoU of final state) + (1 bonus point if IoU == 1)
            
        An empty plan returns 0.

        Returns:
        A float representing the total reward.
        """
        # Split plan lines and filter out any end markers.
        plan_lines = [
            line.strip() for line in self.plan.strip().split("\n")
            if line.strip() and "[plan end]" not in line.lower()
        ]
        if not plan_lines:
            print("Empty plan.")
            return 0.0
        parsed_goal = self.simulate_plan(true_plan)
        # Parse the goal state using the same parsing function.
        # print(self.parse_goal)
        # print(parsed_goal)
        goal_state, goal_hand, _ = self.parse_initial_state(parsed_goal)
        goal_set = {(block, loc) for block, loc in goal_state.items()}

        def compute_iou(state_str: str) -> float:
            # Parse the current state and form its condition set.
            state, hand, _ = self.parse_initial_state(state_str)
            state_set = {(block, loc) for block, loc in state.items()}
            union = goal_set.union(state_set)
            return len(goal_set.intersection(state_set)) / len(union)

        current_state = self.init_state
        last_iou = compute_iou(current_state)
        # print(f"Initial IoU: {last_iou}")
        num_true_actions = len([line.strip() for line in true_plan.strip().split("\n") if line.strip() and "[plan end]" not in line.lower()])
        valid_actions_count = 0
        
        for action in plan_lines:
            try:
                current_state = self.simulate_step(current_state, action)
                valid_actions_count += 1.0
                last_iou = compute_iou(current_state)
                # print(f"After action '{action}', IoU: {last_iou}")
            except Exception as e:
                print(f'Error in parsing action - {e}')
                return 0.0
        
        # If no valid actions were performed, set norm_factor to 0.
        if valid_actions_count == 0:
            norm_factor = 0.0
        else:
            deviation_ratio = abs(valid_actions_count - num_true_actions) / num_true_actions
            norm_factor = 1.0 / (1.0 + deviation_ratio)
        print(f"Valid actions: {valid_actions_count}, True actions: {num_true_actions}, Norm factor: {norm_factor}, IoU: {last_iou}")
        # if last_iou == 1.0:
        #     final_reward = last_iou + norm_factor
        # else:
        #     final_reward = last_iou
        final_reward = float(last_iou == 1.0) * (1 + norm_factor)
        print(f"Final reward: {final_reward}")
        return final_reward

    @classmethod
    def test_from_json(cls, json_file: str):
        """
        Loads a JSON file containing a list of problems.
        Each problem must have keys "init", "goal", and "plan".
        The method loops over the problems, tests the plan, prints results for each,
        and prints the overall accuracy (i.e. how many times the goal was reached).
        """
        with open(json_file, "r") as f:
            problems = json.load(f)
        total = len(problems)
        correct = 0
        for i, prob in enumerate(problems, start=1):
            init_str = prob.get("init", "")
            goal_str = prob.get("goal", "")
            plan_str = prob.get("plan", "")
            instance = cls(init_str, goal_str, plan_str)
            final_state_str, reached, missing = instance.test()
            if reached:
                correct += 1
            print(f"--- Problem {i} ---")
            print("Initial State:", init_str)
            print("Plan:\n", plan_str)
            print("Goal:", goal_str)
            print("Final State:", final_state_str)
            print("Goal Reached?", reached)
            if not reached:
                print("Missing Conditions:", missing)
            # Also print the reward as computed by our RL reward model.
            reward = instance.simulate_plan_with_reward()
            print("Reward:", reward)
            print("\n" + "="*50 + "\n")
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"Accuracy: {accuracy:.2f}% ({correct} out of {total} problems reached the goal)")
    
    @classmethod
    def simplify_state_given_reference(cls, reference_state_str: str, final_state_str: str) -> str:
        """
        Compares a reference state with a final state (both given as descriptive strings)
        and returns a simplified description containing only the blocks whose locations
        have changed. If a block is held, it is reported as "the X block is in hand". For
        other changes, it returns "the X block is on top of the Y block" or "on the table".
        
        The resulting facts are returned in an arbitrary (jumbled) order.
        """
        # Use the existing parsing method to obtain state mappings and hand status.
        ref_state, ref_hand, _ = cls.parse_initial_state(reference_state_str)
        final_state, final_hand, _ = cls.parse_initial_state(final_state_str)
        
        # Collect differences: consider every block appearing in either state.
        diff = {}
        all_blocks = set(ref_state.keys()) | set(final_state.keys())
        for block in all_blocks:
            if ref_state.get(block) != final_state.get(block):
                diff[block] = final_state.get(block)
        
        # If the hand status has changed, record that change.
        if final_hand is not None and final_hand != ref_hand:
            diff[final_hand] = "hand"
        
        # Build descriptive facts for each differing block.
        facts = []
        for block, loc in diff.items():
            if loc == "table":
                facts.append(f"the {cls.denormalize_block_name(block)} is on the table")
            elif loc == "hand":
                facts.append(f"the hand is holding the {cls.denormalize_block_name(block)}")
            else:
                facts.append(f"the {cls.denormalize_block_name(block)} is on top of the {cls.denormalize_block_name(loc)}")
        
        if not facts:
            return ""
        elif len(facts) == 1:
            return facts[0]
        else:
            return ", ".join(facts[:-1]) + " and " + facts[-1]
    
    @classmethod
    def get_possible_actions(cls, state_str: str) -> list:
        """
        Returns a list of all possible actions for the given state (provided as a descriptive string).
        
        When the hand is empty:
        - For each clear block that is on top of another block, returns:
            "unstack the X block from on top of the Y block"
        - For each clear block on the table, returns:
            "pick up the X block"
        
        When the hand is holding a block (say X):
        - For every clear block (Y) different from X, returns:
            "stack the X block on top of the Y block"
        - Also returns:
            "put down the X block"
        """
        # Parse the state using the class's parsing method.
        state, hand, blocks_order = cls.parse_initial_state(state_str)
        actions = []
        
        if hand is None:
            # Hand is empty: generate unstack and pickup actions.
            for block, loc in state.items():
                # Unstack action: block must be on top of something (i.e. not "table" or "hand") and must be clear.
                if loc not in ("table", "hand") and cls.is_clear(block, state, hand):
                    actions.append(f"unstack the {cls.denormalize_block_name(block)} from on top of the {cls.denormalize_block_name(loc)}")
            for block, loc in state.items():
                # Pickup action: block must be on the table and clear.
                if loc == "table" and cls.is_clear(block, state, hand):
                    actions.append(f"pick up the {cls.denormalize_block_name(block)}")
        else:
            # Hand is holding a block: generate stack and putdown actions.
            held = hand
            for block, loc in state.items():
                if block != held and cls.is_clear(block, state, hand):
                    actions.append(f"stack the {cls.denormalize_block_name(held)} on top of the {cls.denormalize_block_name(block)}")
            actions.append(f"put down the {cls.denormalize_block_name(held)}")
        
        return actions
    
    @classmethod
    def states_equal(cls, state_str1: str, state_str2: str) -> tuple:
        """
        Checks if two states are equal by comparing the final state (state_str1) against 
        the expected state (state_str2). It uses the class's parsing methods to obtain the 
        state mappings and hand status, and returns a tuple (equal, differences), where:
        
        - equal: True if state_str1 meets all the conditions in state_str2, otherwise False.
        - differences: a list of strings describing any mismatches.
        
        For example, if the expected state is:
        "the red block is on top of the blue block and the blue block is on top of the orange block"
        
        but the final state does not match these relations, the function will return False along 
        with messages like:
        "the red block should be on top of the blue block"
        """
        # Parse both states.
        state1, hand1, _ = cls.parse_initial_state(state_str1)
        state2, hand2, _ = cls.parse_initial_state(state_str2)
        
        equal = True
        differences = []
        
        # Compare hand status.
        if hand1 != hand2:
            equal = False
            if hand2 is None:
                differences.append("hand should be empty")
            else:
                differences.append(f"the {cls.denormalize_block_name(hand2)} should be in hand")
        
        # For each expected block relation in state2, verify state1 matches.
        for block, expected in state2.items():
            if block not in state1 or state1[block] != expected:
                equal = False
                differences.append(
                    f"the {cls.denormalize_block_name(block)} should be on {cls.denormalize_block_name(expected)}"
                )
        return equal, differences
    
    


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == '__main__':
    # Example 1: Plan Simulation and Goal Testing.
    init1 = ("the red block is clear, the blue block is clear, the orange block is clear, the hand is empty, "
             "the blue block is on top of the yellow block, the yellow block is on the table, "
             "the red block is on the table and the orange block is on the table")
    goal1 = "the red block is on top of the blue block and the blue block is on top of the orange block"
    plan1 = """
unstack the blue block from on top of the yellow block
stack the blue block on top of the orange block
pick up the red block
stack the red block on top of the blue block
[PLAN END]
"""
# stack the red block on top of the blue block
    instance1 = BlocksWorldModel(init1, goal1, plan1)
    final_state_str1, reached1, missing1 = instance1.test()
    print("Example 1:")
    print("Final State:", final_state_str1)
    print("Goal Reached?", reached1)
    if not reached1:
        print("Missing Conditions:", missing1)
    print("\n" + "="*50 + "\n")
    
    # Example 2: Plan Simulation and Goal Testing.
    init2 = ("the C block is clear, the B block is clear, the hand is empty, "
             "the C block is on top of the A block, the A block is on top of the D block, "
             "the D block is on the table and the B block is on the table")
    goal2 = "the A block is on top of the B block"
    plan2 = """
unstack the C block from on top of the A block
put down the C block
unstack the A block from on top of the D block
stack the A block on top of the B block
[PLAN END]
"""
    instance2 = BlocksWorldModel(init2, goal2, plan2)
    final_state_str2, reached2, missing2 = instance2.test()
    print("Example 2:")
    print("Final State:", final_state_str2)
    print("Goal Reached?", reached2)
    if not reached2:
        print("Missing Conditions:", missing2)
    print("\n" + "="*50 + "\n")
    
    # Example 3: Single-Step Simulation.
    init_state_example = "the red block is clear, the hand is empty, the red block is on the table"
    action_example = "pick up the red block"
    new_state = BlocksWorldModel.simulate_step(init_state_example, action_example)
    print("Example 3: Single-Step Simulation")
    print("Initial State:", init_state_example)
    print("Action:", action_example)
    print("New State:", new_state)
    print("\n" + "="*50 + "\n")
    
    # Example 4: Step-by-Step Simulation of an entire plan and goal checking.
    print("Example 4: Step-by-Step Simulation and Goal Check")
    current_state = init1
    plan_lines = [line.strip() for line in plan1.strip().split("\n") if line.strip() and "[plan end]" not in line.lower()]
    for action in plan_lines:
        current_state = BlocksWorldModel.simulate_step(current_state, action)
        print(f"After action '{action}':\n  {current_state}\n")
    # Now parse the final state string into a state dictionary.
    final_state_dict, _, _ = BlocksWorldModel.parse_initial_state(current_state)
    # Parse the goal into a dictionary.
    goal_dict = BlocksWorldModel.parse_goal(goal1)
    # Compare final state against goal.
    goal_reached = True
    missing_conditions = []
    for block, expected in goal_dict.items():
        if block not in final_state_dict or final_state_dict[block] != expected:
            goal_reached = False
            missing_conditions.append(f"{BlocksWorldModel.denormalize_block_name(block)} should be on {BlocksWorldModel.denormalize_block_name(expected)}")
    print('init:', init1)
    print("Final State:", current_state)
    print("Goal:", goal1)
    print("Goal Reached?", goal_reached)
    
        
    reference_state = ("the red block is clear, the blue block is clear, the orange block is clear, the hand is empty, "
                       "the blue block is on top of the yellow block, the yellow block is on the table, "
                       "the red block is on the table and the orange block is on the table")
    goal_state = ("the yellow block is clear, the red block is clear, the hand is empty, the yellow block is on the table, "
                  "the blue block is on top of the orange block, the red block is on top of the blue block and the orange block is on the table")
    
    simplified_goal = BlocksWorldModel.simplify_state_given_reference(init1, current_state)
    print("Simplified Goal:", simplified_goal)
    print(BlocksWorldModel.get_possible_actions(current_state))
    
    print(BlocksWorldModel.states_equal(current_state, simplified_goal))
    
    current_state = 'the 2nd block is clear, the hand is empty, the 3rd block is on top of the 1st block, the 2nd block is on top of the 3rd block and the 1st block is on the table'
    final_s = 'the hand is holding the 2nd block'
    final_s = 'the 3rd block is on top of the 2nd block and the 2nd block is on the table'
    plan_lines = 'unstack the 2nd block from on top of the 3rd block\nput down the 2nd block\nunstack the 3rd block from on top of the 1st block\nstack the 3rd block on top of the 1st block'
    tru_plan = 'unstack the 2nd block from on top of the 3rd block\nput down the 2nd block'
    bw_model = BlocksWorldModel(init_state=current_state, goal=final_s, plan=plan_lines)
    # bw_model.simulate_plan_with_reward(tru_plan)
    print(bw_model.simulate_plan_with_reward(tru_plan))
#     json_path = '/mnt/data/shared/shparashar/Sys2Bench/data/blocksworld/train_set-6.json'
#     BlocksWorldModel.test_from_json(json_path)
    
#     # Example usage of simulate_plan_with_reward (RL style reward).
#     init_example = ("the red block is clear, the blue block is clear, the orange block is clear, the hand is empty, "
#                     "the blue block is on top of the yellow block, the yellow block is on the table, "
#                     "the red block is on the table and the orange block is on the table")
#     goal_example = "the red block is on top of the blue block and the blue block is on top of the orange block"
#     plan_example = """
# unstack the blue block from on top of the yellow block
# stack the blue block on top of the orange block
# pick up the red block
# stack the red block on top of the blue block
# [PLAN END]
# """
#     instance_example = BlocksWorldModel(init_example, goal_example, plan_example)
#     reward = instance_example.simulate_plan_with_reward()
#     print("RL Reward for the example plan:", reward)
    
    
#     # Wrong plan test.
#     init_example = "the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the yellow block is on top of the red block, the red block is on the table and the orange block is on the table"
#     goal_example = "the red block is on top of the blue block and the blue block is on top of the orange block"
#     plan_example = """
# unstack the blue block from on top of the orange block
# stack the blue block on top of the orange block
# unstack the blue block from on top of the orange block
# stack the blue block on top of the red block    
#     """
#     instance_example = BlocksWorldModel(init_example, goal_example, plan_example)
#     reward = instance_example.simulate_plan_with_reward()
#     print("RL Reward for the example plan:", reward)
    
#     plan ="""
# pick up the block 1 
# stack the block 1 on top of the block 2 
# pick up the block 2 
# stack the block 2 on top of the block 3 
# pick up the block 3   
#     """
#     init_example = "the block 3 is clear, the block 1 is clear, the block 2 is clear, the hand is empty, the block 3 is on the table, the block 1 is on the table and the block 2 is on the table"
#     goal_example = "the block 3 is on top of the block 2 and the block 1 is on top of the block 3"
#     instance_example = BlocksWorldModel(init_example, goal_example, plan)
#     print('herer')
#     reward = instance_example.simulate_plan_with_reward()
#     print("RL Reward for the example plan:", reward)