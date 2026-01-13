import random
import re
from collections import Counter

def sc_output_extractor(algo_output, mode = "majority"):
    """
    If algo_output is a flat list, returns the majority extract_plan().
    If algo_output is a list of lists, performs majority vote per index
    and returns a list of results.
    """
    if not algo_output:
        return None

    if mode == "pass":
        answers = [extract_plan(x) for x in algo_output]
        return answers
    
    # Helper to get the top-voted plan from a list of raw outputs
    def majority_plan(raw_outputs):
        answers = [extract_plan(x) for x in raw_outputs]
        counter = Counter(answers)
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    # Detect nested list case
    first = algo_output[0]
    if isinstance(first, list):
        # assume all inner lists are same length
        length = len(first)
        results = []
        for idx in range(length):
            # collect the idx-th element from each sub-list
            column = [sublist[idx] for sublist in algo_output]
            results.append(majority_plan(column))
        return results

    # plain-list case
    return majority_plan(algo_output)


def extract_plan(text):
    # This pattern matches either an <answer> or <plan> tag.
    pattern = r"<(answer|plan)>(.*?)</\1>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # match.group(2) contains the content inside the tag.
        return match.group(2).strip()
    else:
        print("No match found in output:", text)
        return ""

def generate_icl(examples, provide_think_icl=False, num_icl = 2, idx=None):
    
    if idx is not None:
        filtered_examples = [ex for i, ex in enumerate(examples) if i != idx]
    else:
        filtered_examples = examples
    
    sampled_examples = random.sample(filtered_examples, num_icl)
    if num_icl == 1:
        icl_text = "Here is an example problem:\n\n"
    else:
        icl_text = "Here are some example problems:\n\n"
    for example in sampled_examples:
        icl_text += f"Here is the initial state of the blocks: {example['init']}\n\nHere is the goal state of the blocks: {example['goal']}.\nShow your work in the <think> </think> tags. Return the final sequence of actions as the plan in the <plan> </plan> tags.\n"
        if provide_think_icl:
            icl_text += example['trace']
        else:
            icl_text += f"<think>To solve this problem, I need to come up with the right set of actions to achieve the goal.</think>\n<plan>\n{extract_plan(example['trace'])}\n</plan>\n\n"
    return icl_text
