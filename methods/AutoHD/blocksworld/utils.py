import re
from itertools import permutations, product
import importlib

class TimeoutException(Exception):
    """Custom exception for timeouts."""
    pass

def timeout_handler(signum, frame):
    """Handler to raise an exception on timeout."""
    raise TimeoutException("Function execution timed out")

# Helper function to generate actions for the Game of 24, during heuristic search.
# Saves time by not asking the LLM to generate actions.
# Ensures bias free evaluation of heuristic.

# Import libraries or modules.
def build_global_scope(code):
        import_statements = re.findall(r'^\s*(?:import|from\s+\S+\s+import)\s+.+', code, re.MULTILINE)
        global_scope = {}

        for statement in import_statements:
            try:
                # Handle 'import module [as alias]' and 'from module import ... [as alias]'
                if statement.startswith("import"):
                    modules = statement.split("import")[-1].split(",")
                    for module in modules:
                        parts = module.strip().split(" as ")
                        module_name = parts[0].strip()
                        alias = parts[1].strip() if len(parts) > 1 else module_name
                        global_scope[alias] = importlib.import_module(module_name)
                elif statement.startswith("from"):
                    parts = statement.split("import")
                    module_name = parts[0].replace("from", "").strip()
                    imported_names = parts[1].split(",")
                    module = importlib.import_module(module_name)
                    for name in imported_names:
                        alias_parts = name.strip().split(" as ")
                        original_name = alias_parts[0].strip()
                        alias = alias_parts[1].strip() if len(alias_parts) > 1 else original_name
                        global_scope[alias] = getattr(module, original_name)
            except Exception as e:
                print(f"Failed to import {statement}: {e}")

        return global_scope

# Create callable function from code text.
def create_callable_function(code) -> callable:
    local_scope = {}
    global_scope = build_global_scope(code)
    exec(code, global_scope, local_scope)
    return local_scope['calc_heuristic']

def extract_heuristics(text):
    # Step 1: Split by generations
    generation_pattern = r"Generation \d+"
    generations = re.split(generation_pattern, text)
    generation_headers = re.findall(generation_pattern, text)
    
    # Step 2: Extract functions and accuracies from each generation
    generation_data = []
    for i, generation in enumerate(generations[1:]):  # Skip the first empty split
        functions = []
        function_pattern = r"Function:\s*(.*?)\s*Accuracy:\s*([\d.]+)"
        matches = re.findall(function_pattern, generation, re.DOTALL)
        for func, acc in matches:
            functions.append({"function": func.strip(), "accuracy": float(acc)})
        # print(f"Generation {generation_headers[i]}: {len(functions)} functions")
        generation_data.append({"generation": generation_headers[i], "functions": functions})
    
    # Step 3: Find the best function in each generation
    best_per_generation = []
    best_overall = {"generation": None, "function": None, "accuracy": float('-inf')}
    for gen_data in generation_data:
        best_in_gen = max(gen_data["functions"], key=lambda x: x["accuracy"], default=None)
        if best_in_gen:
            best_per_generation.append({"generation": gen_data["generation"], **best_in_gen})
            if best_in_gen["accuracy"] > best_overall["accuracy"]:
                best_overall = {"generation": gen_data["generation"], **best_in_gen}
    return generation_data, best_per_generation, best_overall


def make_actions(state):
    if ', and ' in state:
        state = re.sub(r',\s+and\s+', ', ', state, count=1)
    if ' and ' in state:
        state = re.sub(r'\s+and\s+', ', ', state, count=1)
    # print(state)
    state = state.lower().strip().strip(".").split(", ")
    actions = []
    colors = []
    # print(state)

    for s in state:
            if 'clear' in s:
                color = re.findall(r"the (\w+) block",s)[0]
                assert s== f"the {color} block is clear", f"color is {color} and s is {s}"
                colors.append(color)
    
    assert len(colors)>0
    # print(colors)
    if "the hand is empty" in state:
        for color in colors:
            state_covered = False
            for s in state:
                if s==f"the {color} block is clear":
                    continue
                elif f"the {color} block is on top of the" in s:
                    color2 = re.findall(rf"the {color} block is on top of the (\w+) block",s)[0]
                    actions.append(f"unstack the {color} block from on top of the {color2} block")
                    state_covered = True
                elif s==f"the {color} block is on the table":
                    actions.append(f"pick up the {color} block")
                    state_covered = True
                if state_covered:
                    break
            if not state_covered:
                print("-----------")
                print(s)
                print(color)
                assert False, f"color is {color}"
            
            
    else:
        hand_color=''
        for s in state:
            if 'the hand is holding the' in s:
                hand_color = re.findall(r"the (\w+) block",s)[0]
                break
        assert hand_color!=''

        for color in colors:
            actions.append(f"stack the {hand_color} block on top of the {color} block")
        actions.append(f"put down the {hand_color} block")

    return actions