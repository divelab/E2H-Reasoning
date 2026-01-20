import re
from math_verify import (
    parse, 
    verify,
    LatexExtractionConfig,
    ExprExtractionConfig,
    StringExtractionConfig
)


def is_equiv(completion, answer):
    answer_match = re.findall(r'<answer>\s*(.*?)\s*</answer>', completion, re.DOTALL)
    if len(answer_match) == 0:
        return False
    pred = answer_match[-1].strip()
    if pred==answer:
        return True
    else:
        return verify(
            parse(
                answer,
                extraction_config = [
                    LatexExtractionConfig(),
                    ExprExtractionConfig(),
                    StringExtractionConfig(
                        strings=("A", "B", "C", "D", "E", "F")
                    )
                ]
            ), 
            parse(
                pred,
                extraction_config = [
                    LatexExtractionConfig(),
                    ExprExtractionConfig(),
                    StringExtractionConfig(
                        strings=("A", "B", "C", "D", "E", "F")
                    )
                ]
            )
        )


def last_boxed_only_string(string: str) -> str:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]