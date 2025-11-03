import re
import sympy
import logging
from sympy.parsing.latex import parse_latex
from functools import partial, update_wrapper
import signal
import json

def get_reward_fn(
    cfg
):

    reward_classes = globals()
    assert cfg.format_reward_fn in reward_classes
    assert cfg.correctness_reward_fn in reward_classes
       
    reward_fn = partial(
        base_reward_fn,
        format_reward=cfg.format_reward,
        format_reward_fn=globals()[cfg.format_reward_fn](), 
        correctness_reward=cfg.correctness_reward,
        correctness_reward_fn=globals()[cfg.correctness_reward_fn]()
    )
    update_wrapper(reward_fn, base_reward_fn)

    return reward_fn


def base_reward_fn(
    format_reward, 
    format_reward_fn,
    correctness_reward, 
    correctness_reward_fn,
    **kwargs
):
    
    kwargs.pop('trainer_state', None)
    inputs_per_sample = [dict(zip(kwargs, values)) for values in zip(*kwargs.values())]
    
    rewards = []
    for sample in inputs_per_sample:

        try:
            reward = 0.0
            if format_reward_fn("<think>" + sample['completions']):
                reward += format_reward
                predicted_answer = re.findall(r'<answer>\s*(.*?)\s*</answer>', sample['completions'], re.DOTALL)[-1].strip()   
                
                result = correctness_reward_fn(predicted_answer, **sample)
                if isinstance(result, bool) and result: # Works for countdown and arithmetic tasks.
                    reward += correctness_reward
                else:
                    reward += result
                # if correctness_reward_fn(predicted_answer, **sample):
                #     reward += correctness_reward
            rewards.append(reward)

        except Exception as e:
            rewards.append(0.0)

    return rewards

class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class FormatReward:
    def __call__(self, response):
        response = response.strip()

        # Rule 1: Must start with <think> and end with </answer>
        if not response.startswith("<think>") or not response.endswith("</answer>"):
            return False

        # Rule 2: Must contain exactly one of each tag.
        if response.count("<think>") != 1 or response.count("</think>") != 1:
            return False
        if response.count("<answer>") != 1 or response.count("</answer>") != 1:
            return False

        # Find indices for each tag.
        think_open = response.find("<think>")
        think_close = response.find("</think>")
        plan_open = response.find("<answer>")
        plan_close = response.find("</answer>")

        # Rule 3: The order should be: <think> ... </think> then <answer> ... </answer>
        if think_open != 0:  # Should start with <think>
            return False
        if (think_close==-1) or (plan_open==-1) or (plan_close==-1):
            return False
        if think_close > plan_open:
            return False

        # Rule 4: Check non-empty content between tags.
        think_content = response[len("<think>"):think_close].strip()
        plan_content = response[plan_open + len("<answer>"):plan_close].strip()
        if not think_content or not plan_content:
            return False

        # Rule 5: Check <answer> immedietly follows </think>
        if not (response[think_close+len("</think>"):plan_open].strip() == ''):
            return False

        return True


class AquaCorrectnessReward:
    def __call__(self, predicted_answer, **kwargs):
        true_answer = kwargs['answer']
        return predicted_answer==true_answer


class Gsm8kCorrectnessReward:
    def __call__(self, predicted_answer, **kwargs):
        true_answer = kwargs['answer']
        predicted_answer = predicted_answer.replace(",", "")
        predicted_answer = re.search(r'-?\d+(?:\.\d+)?', predicted_answer)
        if predicted_answer is not None:
            predicted_answer = float(predicted_answer.group(0))
            return abs(true_answer - predicted_answer) < 1e-5
        return False


class MathCorrectnessReward:

    INVALID_ANSWER = "[invalidanswer]"
    SUBSTITUTIONS = [
        ("an ", ""),
        ("a ", ""),
        (".$", "$"),
        ("\\$", ""),
        (r"\ ", ""),
        (" ", ""),
        ("mbox", "text"),
        (",\\text{and}", ","),
        ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"),
    ]
    REMOVED_EXPRESSIONS = [
        "square",
        "ways",
        "integers",
        "dollars",
        "mph",
        "inches",
        "ft",
        "hours",
        "km",
        "units",
        "\\ldots",
        "sue",
        "points",
        "feet",
        "minutes",
        "digits",
        "cents",
        "degrees",
        "cm",
        "gm",
        "pounds",
        "meters",
        "meals",
        "edges",
        "students",
        "childrentickets",
        "multiples",
        "\\text{s}",
        "\\text{.}",
        "\\text{\ns}",
        "\\text{}^2",
        "\\text{}^3",
        "\\text{\n}",
        "\\text{}",
        r"\mathrm{th}",
        r"^\circ",
        r"^{\circ}",
        r"\;",
        r",\!",
        "{,}",
        '"',
        "\\dots",
    ]

    def __call__(self, predicted_answer, **kwargs):
        true_answer = kwargs['answer']
        predicted_answer = self.normalize_final_answer(predicted_answer)
        true_answer = self.normalize_final_answer(true_answer)
        if predicted_answer == self.INVALID_ANSWER:
            return False
        if (predicted_answer.strip()==true_answer.strip()) or self.is_equiv(true_answer, predicted_answer):
            return True
        else:
            return False

    def normalize_final_answer(self, final_answer):
        """
        Normalize a final answer to a quantitative reasoning question.

        Copied character for character from appendix D of Lewkowycz et al. (2022)
        """
        final_answer = final_answer.split("=")[-1]

        for before, after in self.SUBSTITUTIONS:
            final_answer = final_answer.replace(before, after)
        for expr in self.REMOVED_EXPRESSIONS:
            final_answer = final_answer.replace(expr, "")

        # Extract answer that is in LaTeX math, is bold,
        # is surrounded by a box, etc.
        final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
        final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

        # Normalize shorthand TeX:
        #  \fracab -> \frac{a}{b}
        #  \frac{abc}{bef} -> \frac{abc}{bef}
        #  \fracabc -> \frac{a}{b}c
        #  \sqrta -> \sqrt{a}
        #  \sqrtab -> sqrt{a}b
        final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
        final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
        final_answer = final_answer.replace("$", "")

        # Normalize 100,000 -> 100000
        if final_answer.replace(",", "").isdigit():
            final_answer = final_answer.replace(",", "")

        return final_answer

    def is_equiv(self, x1, x2):
        """
        x1 and x2 are normalized latex string
        """
        eval_logger = logging.getLogger(__name__)
        try:
            with timeout(seconds=1):
                try:
                    parsed_x1 = parse_latex(x1)
                    parsed_x2 = parse_latex(x2)
                except (
                    sympy.parsing.latex.errors.LaTeXParsingError,
                    sympy.SympifyError,
                    TypeError,
                ):
                    eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                    return False

                try:
                    diff = parsed_x1 - parsed_x2
                except TypeError:
                    eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                    return False

                try:
                    if sympy.simplify(diff) == 0:
                        return True
                    else:
                        return False
                except ValueError:
                    eval_logger.debug(
                        f"Had some trouble simplifying when comparing {x1} and {x2}"
                    )
        except TimeoutError:
            eval_logger.debug(f"Timed out comparing {x1} and {x2}")
            return False
        except ImportError as e:
            eval_logger.error(e)
            raise
        except Exception as e:
            eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
            return False


class CountdownCorrectnessReward:
    def __call__(self, predicted_answer, **kwargs):
        nums = kwargs['question']
        target = kwargs['answer']
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', predicted_answer)]
        # Each number should be used exactly once
        if sorted(numbers_in_eq) == sorted(nums):
            # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
            if re.match(r'^[\d+\-*/().\s]+$', predicted_answer):
                # Evaluate the equation with restricted globals and locals
                result = eval(predicted_answer, {"__builtins__": None}, {})
                # Account for floating point precision
                if abs(result - target) < 1e-5:
                    return True
        return False

class BlocksworldCorrectnessReward:
    @staticmethod
    def normalize_input_string(s: str) -> str:
        return re.sub(r'\bblock (\d+)\b', r'\1 block', s, flags=re.IGNORECASE)
    
    @staticmethod
    def normalize_block_name(name: str) -> str:
        name = name.strip().lower()
        if name.endswith(" block"):
            name = name[:-6].strip()
        return name
    
    @staticmethod
    def denormalize_block_name(name: str) -> str:
        return f"{name} block"
    
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
                r"^unstack the (\w+) block from on top of the (\w+) block[^\w\s]*\s*$",
                lambda m: ("unstack", cls.normalize_block_name(m.group(1)), cls.normalize_block_name(m.group(2)))
            ),
            (
                r"^pick up the (\w+) block[^\w\s]*\s*$",
                lambda m: ("pickup", cls.normalize_block_name(m.group(1)))
            ),
            (
                r"^stack the (\w+) block on top of the (\w+) block[^\w\s]*\s*$",
                lambda m: ("stack", cls.normalize_block_name(m.group(1)), cls.normalize_block_name(m.group(2)))
            ),
            (
                r"^put down the (\w+) block[^\w\s]*\s*$",
                lambda m: ("putdown", cls.normalize_block_name(m.group(1)))
            ),
        ]
        for pattern, action_fn in patterns:
            m = re.match(pattern, action_str)
            if m:
                return action_fn(m)
        raise ValueError("Action not recognized or unsupported.")
    

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
    def simulate_plan(cls, init_state, plan=None, simplify=False):
        """
        Simulates the sequence of actions in the plan starting from the initial state.
        Returns the final internal state, hand, blocks_order, and a description string.
        """
        lines = [line.strip() for line in plan.strip().split("\n")
                 if line.strip() and "[plan end]" not in line.lower()]
        state = init_state
        for action_line in lines:
            state = cls.simulate_step(state, action_line)
        final_state_str = state
        if simplify:
            final_state_str = cls.simplify_state_given_reference(init_state, final_state_str)
        return final_state_str
    
    @classmethod
    def states_equal(cls, state_str1: str, state_str2: str) -> tuple:
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
    
    @classmethod
    def check_goal(cls, state: str, goal: str):
        return cls.states_equal(state, goal)
    
    
    @classmethod
    def simulate_plan_with_reward(cls, init_state: str, predicted_plan: str, true_plan: str) -> float:
        # Split plan lines and filter out any end markers.
        plan_lines = [
            line.strip() for line in predicted_plan.strip().split("\n")
            if line.strip() and "[plan end]" not in line.lower()
        ]
        if not plan_lines:
            print("Empty plan.")
            return 0.0
        parsed_goal = cls.simulate_plan(init_state=init_state, plan=true_plan)
        goal_state, goal_hand, _ = cls.parse_initial_state(parsed_goal)
        goal_set = {(block, loc) for block, loc in goal_state.items()}

        def compute_iou(state_str: str) -> float:
            # Parse the current state and form its condition set.
            state, hand, _ = cls.parse_initial_state(state_str)
            state_set = {(block, loc) for block, loc in state.items()}
            union = goal_set.union(state_set)
            return len(goal_set.intersection(state_set)) / len(union)

        current_state = init_state
        last_iou = compute_iou(current_state)
        num_true_actions = len([line.strip() for line in true_plan.strip().split("\n") if line.strip() and "[plan end]" not in line.lower()])
        valid_actions_count = 0
        
        for action in plan_lines:
            try:
                current_state = cls.simulate_step(current_state, action)
                valid_actions_count += 1.0
                last_iou = compute_iou(current_state)
            except Exception as e:
                print(f'Error in parsing action - {e}')
                return 0.0
        
        # If no valid actions were performed, set norm_factor to 0.
        if valid_actions_count == 0:
            norm_factor = 0.0
        elif valid_actions_count <= num_true_actions:
            norm_factor = 1.0
        else:
            deviation_ratio = (valid_actions_count - num_true_actions) / num_true_actions
            norm_factor = 1.0 / (1.0 + deviation_ratio)
        final_reward = float(last_iou == 1.0) * (1 + norm_factor)
        print(f"Final reward: {final_reward}")
        return final_reward
    
    @classmethod
    def __call__(cls, predicted_plan, **kwargs):
        try:
            init_state = kwargs['question']
            _ = kwargs['answer'] # Goal State 
            true_plan = kwargs['solution']
            reward = cls.simulate_plan_with_reward(init_state=init_state, predicted_plan=predicted_plan, true_plan=true_plan)
            return reward
        except Exception as e:
            print(f'Error in computing BW verifier reward - {e}')
            return 0.0