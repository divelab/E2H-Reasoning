from reasoners.algorithm import HeuristicGuidedSearchResult
from world_model import Game24State
import re
from itertools import permutations, product
import importlib

class TimeoutException(Exception):
    """Custom exception for timeouts."""
    pass

def timeout_handler(signum, frame):
    """Handler to raise an exception on timeout."""
    raise TimeoutException("Function execution timed out")

def parse_result(result: list[HeuristicGuidedSearchResult]):
    expressions = []
    for terminal_state in result:
        game24State: Game24State = terminal_state.terminal_state
        if game24State.expression and '=' in game24State.expression:
            expressions.append(game24State.expression.split('=')[0])
    if len(expressions) > 0:
        return expressions
    return None

# Helper function to generate actions for the Game of 24, during heuristic search.
# Saves time by not asking the LLM to generate actions.
# Ensures bias free evaluation of heuristic.
def generate_actions(numbers):
    operators = ['+', '-', '*', '/']
    actions = []

    def calculate_action(operand1, operand2, operator):
        try:
            if operator == '+':
                return operand1 + operand2
            elif operator == '-':
                return operand1 - operand2
            elif operator == '*':
                return operand1 * operand2
            elif operator == '/':
                if operand2 == 0:  # Avoid division by zero
                    return None
                return operand1 / operand2
        except ZeroDivisionError:
            return None

    action_number = 1
    for perm in permutations(numbers, len(numbers)):
        for i in range(len(numbers) - 1):
            for operator in operators:
                operand1 = perm[i]
                operand2 = perm[i + 1]
                result = calculate_action(operand1, operand2, operator)
                if result is not None:
                    remaining_numbers = list(perm[:i] + perm[i+2:] + (result,))
                    action = f"{operand1} {operator} {operand2} = {result} (left: {' '.join(map(str, remaining_numbers))})"
                    actions.append(action)
                    action_number += 1

    return ["\n".join(actions)]

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