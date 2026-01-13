import numpy as np
import re
import importlib
from reasoners import Evaluator

def getCube(s):
    cube_string = ""
    cube_string += "Upper:\n"  
    cube_string += "{} {}\n".format(s[0], s[1])  
    cube_string += "{} {}\n".format(s[2], s[3])  

    cube_string += "Front:\n"  
    cube_string += "{} {}\n".format(s[8], s[9])  
    cube_string += "{} {}\n".format(s[10], s[11])   

    cube_string += "Down:\n"  
    cube_string += "{} {}\n".format(s[12], s[13])  
    cube_string += "{} {}\n".format(s[14], s[15]) 

    cube_string += "Left:\n"  
    cube_string += "{} {}\n".format(s[16], s[17])  
    cube_string += "{} {}\n".format(s[18], s[19])   

    cube_string += "Right:\n"  
    cube_string += "{} {}\n".format(s[4], s[5])  
    cube_string += "{} {}\n".format(s[6], s[7]) 

    cube_string += "Back:\n"  
    cube_string += "{} {}\n".format(s[20], s[21])  
    cube_string += "{} {}\n".format(s[22], s[23])  

    return cube_string

def parseCube(cube_string):
    # Initialize the array
    s = []

    # Define patterns to match each face (case-insensitive)
    patterns = {
        "Upper": r"(?i)upper:\s*(\d)\s+(\d)\s*(\d)\s+(\d)",
        "Right": r"(?i)right:\s*(\d)\s+(\d)\s*(\d)\s+(\d)",
        "Front": r"(?i)front:\s*(\d)\s+(\d)\s*(\d)\s+(\d)",
        "Down": r"(?i)down:\s*(\d)\s+(\d)\s*(\d)\s+(\d)",
        "Left": r"(?i)left:\s*(\d)\s+(\d)\s*(\d)\s+(\d)",
        "Back": r"(?i)back:\s*(\d)\s+(\d)\s*(\d)\s+(\d)"
    }

    # Follow the order: Upper, Right, Front, Down, Left, Back
    order = ["Upper", "Right", "Front", "Down", "Left", "Back"]
    
    # Loop through each face in the correct order and extract numbers
    for face in order:
        pattern = patterns[face]
        match = re.search(pattern, cube_string)
        if match:
            # Append all 4 numbers in the order they appear
            s.extend(map(int, match.groups()))
        else:
            raise ValueError(f"Could not parse the {face} face. Check the format.")
    
    return np.array(s)

def isSolved(s):
    for i in range(6):
        # 检查每个面的四个元素是否相同
        if not all(x == s[4 * i] for x in s[4 * i:4 * i + 4]):
            return False
    return True

# move definitions
moveDefs = np.array([ \
  [  2,  0,  3,  1, 20, 21,  6,  7,  4,  5, 10, 11, 12, 13, 14, 15,  8,  9, 18, 19, 16, 17, 22, 23], \
  [  1,  3,  0,  2,  8,  9,  6,  7, 16, 17, 10, 11, 12, 13, 14, 15, 20, 21, 18, 19,  4,  5, 22, 23], \
  [  3,  2,  1,  0, 16, 17,  6,  7, 20, 21, 10, 11, 12, 13, 14, 15,  4,  5, 18, 19,  8,  9, 22, 23], \
  [  0,  9,  2, 11,  6,  4,  7,  5,  8, 13, 10, 15, 12, 22, 14, 20, 16, 17, 18, 19,  3, 21,  1, 23], \
  [  0, 22,  2, 20,  5,  7,  4,  6,  8,  1, 10,  3, 12,  9, 14, 11, 16, 17, 18, 19, 15, 21, 13, 23], \
  [  0, 13,  2, 15,  7,  6,  5,  4,  8, 22, 10, 20, 12,  1, 14,  3, 16, 17, 18, 19, 11, 21,  9, 23], \
  [  0,  1, 19, 17,  2,  5,  3,  7, 10,  8, 11,  9,  6,  4, 14, 15, 16, 12, 18, 13, 20, 21, 22, 23], \
  [  0,  1,  4,  6, 13,  5, 12,  7,  9, 11,  8, 10, 17, 19, 14, 15, 16,  3, 18,  2, 20, 21, 22, 23], \
  [  0,  1, 13, 12, 19,  5, 17,  7, 11, 10,  9,  8,  3,  2, 14, 15, 16,  6, 18,  4, 20, 21, 22, 23], \
  [  0,  1,  2,  3,  4,  5, 10, 11,  8,  9, 18, 19, 14, 12, 15, 13, 16, 17, 22, 23, 20, 21,  6,  7], \
  [  0,  1,  2,  3,  4,  5, 22, 23,  8,  9,  6,  7, 13, 15, 12, 14, 16, 17, 10, 11, 20, 21, 18, 19], \
  [  0,  1,  2,  3,  4,  5, 18, 19,  8,  9, 22, 23, 15, 14, 13, 12, 16, 17,  6,  7, 20, 21, 10, 11], \
  [ 23,  1, 21,  3,  4,  5,  6,  7,  0,  9,  2, 11,  8, 13, 10, 15, 18, 16, 19, 17, 20, 14, 22, 12], \
  [  8,  1, 10,  3,  4,  5,  6,  7, 12,  9, 14, 11, 23, 13, 21, 15, 17, 19, 16, 18, 20,  2, 22,  0], \
  [ 12,  1, 14,  3,  4,  5,  6,  7, 23,  9, 21, 11,  0, 13,  2, 15, 19, 18, 17, 16, 20, 10, 22,  8], \
  [  5,  7,  2,  3,  4, 15,  6, 14,  8,  9, 10, 11, 12, 13, 16, 18,  1, 17,  0, 19, 22, 20, 23, 21], \
  [ 18, 16,  2,  3,  4,  0,  6,  1,  8,  9, 10, 11, 12, 13,  7,  5, 14, 17, 15, 19, 21, 23, 20, 22], \
  [ 15, 14,  2,  3,  4, 18,  6, 16,  8,  9, 10, 11, 12, 13,  1,  0,  7, 17,  5, 19, 23, 22, 21, 20], \
  [  8,  9, 10, 11,  6,  4,  7,  5, 12, 13, 14, 15, 23, 22, 21, 20, 17, 19, 16, 18,  3,  2,  1,  0], \
  [ 23, 22, 21, 20,  5,  7,  4,  6,  0,  1,  2,  3,  8,  9, 10, 11, 18, 16, 19, 17, 15, 14, 13, 12], \
  [ 12, 13, 14, 15,  7,  6,  5,  4, 23, 22, 21, 20,  0,  1,  2,  3, 19, 18, 17, 16, 11, 10,  9,  8], \
  [  2,  0,  3,  1, 20, 21, 22, 23,  4,  5,  6,  7, 13, 15, 12, 14,  8,  9, 10, 11, 16, 17, 18, 19], \
  [  1,  3,  0,  2,  8,  9, 10, 11, 16, 17, 18, 19, 14, 12, 15, 13, 20, 21, 22, 23,  4,  5,  6,  7], \
  [  3,  2,  1,  0, 16, 17, 18, 19, 20, 21, 22, 23, 15, 14, 13, 12,  4,  5,  6,  7,  8,  9, 10, 11], \
  [ 18, 16, 19, 17,  2,  0,  3,  1, 10,  8, 11,  9,  6,  4,  7,  5, 14, 12, 15, 13, 21, 23, 20, 22], \
  [  5,  7,  4,  6, 13, 15, 12, 14,  9, 11,  8, 10, 17, 19, 16, 18,  1,  3,  0,  2, 22, 20, 23, 21], \
  [ 15, 14, 13, 12, 19, 18, 17, 16, 11, 10,  9,  8,  3,  2,  1,  0,  7,  6,  5,  4, 23, 22, 21, 20]  \
])

# move indices
moveInds = { \
  "U": 0, "U'": 1, "U2": 2, "R": 3, "R'": 4, "R2": 5, "F": 6, "F'": 7, "F2": 8, \
  "D": 9, "D'": 10, "D2": 11, "L": 12, "L'": 13, "L2": 14, "B": 15, "B'": 16, "B2": 17, \
  "x": 18, "x'": 19, "x2": 20, "y": 21, "y'": 22, "y2": 23, "z": 24, "z'": 25, "z2": 26 \
}

# apply a move to a state
def doMove(s, move):
  # print(s)
  # print('move',move)
  # print('move2',moveDefs[move])
  return s[moveDefs[move]]

# apply a string sequence of moves to a state
def doAlgStr(s, alg):
  # print('',alg)
  moves = alg.split(" ")
  # print('moves',moves)
  for m in moves:
    if m in moveInds:
      s = doMove(s, moveInds[m])
  return s

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

class TimeoutException(Exception):
    """Custom exception for timeouts."""
    pass

def timeout_handler(signum, frame):
    """Handler to raise an exception on timeout."""
    raise TimeoutException("Function execution timed out")

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

def output_extractor(algo_output):
    try:
        return " ".join(algo_output[0].terminal_node.state.action_history)
    except Exception as e:
        print("Error in output extraction,", e)
        return ""
    

class CubeEvaluator(Evaluator):
    def __init__(self, 
                 data_path,
                 init_prompt,
                 disable_log=False,
                 disable_tqdm=False,
                 output_extractor=output_extractor,
                 answer_extractor=lambda x:x,
                 sample_prompt_type=None) -> None:

        super().__init__()
        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.input_processor = getCube
        import pandas as pd
        test_data =  pd.read_csv(data_path) #pd.read_csv('./data/cube_test.csv')
        convert_to_int = lambda lst: np.array(list(map(lambda x: int(x), lst)))
        test_data['init_state'] = test_data.apply(lambda row: ' '.join(map(str, row[['state_{}'.format(i) for i in range(1, 25)]])), axis=1)
        self.full_dataset = []
        for index, row in test_data.iterrows():
            INIT = ' '.join(map(str, row[['state_{}'.format(i) for i in range(1, 25)]]))
            self.full_dataset.append(convert_to_int(INIT.split()))
        
        self._dataset_name = 'cube'
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.sample_prompt_type = sample_prompt_type


    def sample_prompt(self,
                      shuffle_prompt=True,
                      num_shot=4):
        
        return None
    
    def sample_automatic_prompt(self,
                      shuffle_prompt=True,
                      num_shot=4):   
        
        raise NotImplementedError

    def eval_output(self, answer, output):
        try:
            correct = isSolved(doAlgStr(answer, output))
        except:
            correct = False
        return correct

