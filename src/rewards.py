import re
import sympy
import logging
from sympy.parsing.latex import parse_latex
from functools import partial, update_wrapper


def get_reward_fn(
    cfg
):

    reward_classes = globals()
    assert cfg.format_reward_fn in reward_classes
    assert cfg.correctness_reward_fn in reward_classes
       
    reward_fn = partial(
        base_reward_fn,
        format_reward=cfg.format_reward,
        format_reward_fn=globals()[cfg.format_reward_fn], 
        correctness_reward=cfg.correctness_reward,
        correctness_reward_fn=globals()[cfg.correctness_reward_fn]
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
                if correctness_reward_fn(predicted_answer, **sample):
                    reward += correctness_reward
            rewards.append(reward)

        except Exception as e:
            print(e)
            rewards.append(0.0)

    return rewards


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

