import json
from reasoners.lm.openai_model import OpenAIModel
from reasoners.lm import LLaMaApiModel, HFModel
from inference import autohd_search
from typing import Sequence, Any, Literal, Optional
import fire
import heapq
import random
from action_generation_prompts import get_next_actions_empty, get_next_actions_holding
import numpy as np
import signal
from heuristic_search_prompts  import init_prompt, evolution_prompt, types, evolution_type, dummystate, dummygoal
from utils import timeout_handler, TimeoutException, create_callable_function
import re

class HeuristicSearch:
  def __init__(self, base_model, temperature=0.8, init_prompt=init_prompt, evolution_prompt=evolution_prompt, val_file_path=None, max_generations=5, sample_k=3, log_dir = None, test_file_path = None):
    self.base_model = base_model
    self.init_prompt = init_prompt
    self.n_candidate = 4
    self.temperature = temperature
    self.val_file_path = val_file_path
    self.test_file_path = test_file_path
    self.heuristic_functions = {}
    self.generation = 0
    self.heuristic_functions[self.generation] = []
    self.best_heuristics = {}
    self.best_heuristics[self.generation] = []
    self.max_generations = max_generations
    self.sample_k = sample_k
    self.log_dir = log_dir
    
  def get_initial_heuristics(self, sampled_heuristics=None):

    print("ENTERED GET INITIAL HEURISTICS --- CHECKPOINT!!!")
    print()
    print()
    while len(self.heuristic_functions[self.generation]) < self.sample_k:
      for i in range(4):
        prompts = self.init_prompt
        
        if self.generation == 0:
            pass
        else:
            if i>=4:
              break
            if i<2:
              example_heuristic_prompts = self.prompt_with_sampled_heuristics(sampled_heuristics)
              prompts = evolution_prompt.replace('<exisiting_heuristics>', example_heuristic_prompts)
            else:
              example_heuristic_prompts = self.prompt_with_sampled_heuristics([random.choice(sampled_heuristics)])
              prompts = evolution_prompt.replace('<exisiting_heuristics>', example_heuristic_prompts)
            prompts = prompts.replace('<evolution_type>', evolution_type[types[i]])
        
        print("HEURISTIC GENERATOR PROMPT: ", prompts, flush=True)
        print("--------================-------------====================== ------------------",flush=True)
        # prompts = self.init_prompt.replace('<problem>', example['prompt_0shot'])
        llm_outputs = self.base_model.generate([prompts],
                                  num_return_sequences=self.n_candidate,
                                  #max_length=20,
                                  # eos_token_id=["\n[", "\n", ],
                                  temperature=self.temperature,
                                  do_sample=True,
                                  hide_input=True,
                                  ).text
        for i in range(self.n_candidate):
          # print("llm_outputs[i]  ",llm_outputs[i], flush=True)
          parsed_output = self.parse_output(llm_outputs[i])
          if parsed_output is not None:
            heuristic_function_obj = {'description': parsed_output[0], 'function_text': parsed_output[1], 'llm_output': llm_outputs[i]}
            try:
              self.get_callable_function(i, heuristic_function_obj)
              # print(f"New Heuristic {i}: ",heuristic_function_obj['function_text'])
            except:
              continue
        
        if self.generation == 0:
          break

    print(f'Initial {len(self.heuristic_functions[self.generation])} Heuristics collected.')

    print("--------================-------------====================== ------------------",flush=True)
    # print(self.heuristic_functions[self.generation],flush=True)
    print("--------================-------------====================== ------------------",flush=True)
    
  def get_callable_function(self, id, heuristic_function_obj):
    code = heuristic_function_obj['function_text']
    heuristic_fn = create_callable_function(code=code)
    
    try:
        # Set a timeout for the heuristic function call
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # Timeout duration in seconds
        
        heuristic_fn(dummystate, dummygoal)
    except TimeoutException:
      print(f"Failed to execute heuristic function: Timeout occurred.")
      return
    except Exception as e:
        print(f"Failed to execute heuristic function: Error - {e}")
        return
    finally:
        signal.alarm(0)
        
    print(f"Generation: {self.generation}, Valid Heuristic {id+1}: \n------------\n", code)
    heuristic_function_obj['callable_heuristic'] = heuristic_fn
    self.heuristic_functions[self.generation].append(heuristic_function_obj)
        
  
  def parse_output(self, output: str):
    # Account for parsing errors.
    print('***** Unparsed Output *****')
    print(output)
    try:
      if output.startswith('Heuristic Description:'):
        description_split = output.split("Heuristic Description:", 1)
        
        if len(description_split) < 2 or '```python' not in output:
          print('Python not in op.')
          return None # Code was not parsed.
        description_part, code_part = description_split[1].split('```python', 1)
    
        code_part = code_part.strip('```').strip()
    
        description_part = description_part.strip()
      elif output.startswith('```'):
        match = re.search(r"# Heuristic Description:.*", output)
        if match is None:
          description_part = '' # No description.
        else:
          description_part = match.group(0).replace("# Heuristic Description:", "").strip()
        code_part = output.replace('```','').replace('python', '').strip().strip('```') # Extract the code.
      else:
        print('Does not follow either pattern. Setting as code only.')
        description_part = '' # Description not parsed.
        code_part = output.strip() # Extract the code, it might run.
      return description_part, code_part
    except Exception as e:
      print('Error parsing output',e)
      return None
  
  def prompt_with_sampled_heuristics(self, sampled_heuristics):
    assert self.generation > 0 and sampled_heuristics is not None
    prompt = f'I have {self.sample_k} existing heuristics with their codes as follows:\n'
    for h_i, sample in enumerate(sampled_heuristics):
      _, _, heuristic_obj = sample
      prompt += f'No.{h_i+1} Heuristic Description: {heuristic_obj["description"]}\nCode: {heuristic_obj["function_text"]}\n\n'
    
    return prompt
  
  def test_heuristic(self, prompt, depth_limit=20, temperature=0.8, disable_log='False',lm_plan_file: str = 'lm_plan.tmp', domain_file: str = "data/blocksworld/generated_domain.pddl", config_file: str = "data/blocksworld/bw_config.yaml", **kwargs):
    
    print("Best heuristic in the last generation: ")
    prompt['next_actions_holding'] = get_next_actions_holding(prompt)
    prompt['next_actions_empty'] = get_next_actions_empty(prompt)
    best_item = max(self.best_heuristics[self.max_generations - 1], key=lambda x: x[0])
    best_heuristic_obj = best_item[2]
    
    accuracy = autohd_search(self.base_model, prompt,
                                    disable_log=disable_log, data_path=self.test_file_path, config_file=config_file, domain_file=domain_file, lm_plan_file=lm_plan_file, step_into_state=True, action_prompt=True,
                                    depth_limit=depth_limit, temperature=temperature, 
                                    heuristic_fn=best_heuristic_obj['callable_heuristic'], **kwargs)
    
    print("Function: ")
    print(best_heuristic_obj['function_text'])
    print(f"Accuracy is {accuracy}")
    
    with open(self.log_dir, 'a') as file:
        file.write(f"Test the best heuristic in the last generation: \n")
        file.write(f"Function: \n")
        file.write(best_heuristic_obj['function_text'])
        file.write(f"\n")
        file.write(f"Accuracy: {accuracy} \n")
    
    
    print("Best heuristic in all generations: ")
    prompt['next_actions_holding'] = get_next_actions_holding(prompt)
    prompt['next_actions_empty'] = get_next_actions_empty(prompt)
    all_items = [item for sublist in self.best_heuristics.values() for item in sublist]
    # Find the tuple with the highest fitness_score
    best_item = max(all_items, key=lambda x: x[0])
    # Extract the heuristic_obj
    best_heuristic_obj = best_item[2]
    accuracy = autohd_search(self.base_model, prompt, 
                                    disable_log=disable_log, data_path=self.test_file_path, config_file=config_file, domain_file=domain_file, lm_plan_file=lm_plan_file, step_into_state=True, action_prompt=True,
                                    depth_limit=depth_limit, temperature=temperature, 
                                    heuristic_fn=best_heuristic_obj['callable_heuristic'], **kwargs)
    
    
    print("Function: ")
    print(best_heuristic_obj['function_text'])
    print(f"Accuracy is {accuracy}")
    
    with open(self.log_dir, 'a') as file:
        file.write(f"Test the best heuristic in all the generations: \n")
        file.write(f"Function: \n")
        file.write(best_heuristic_obj['function_text'])
        file.write(f"\n")
        file.write(f"Accuracy: {accuracy} \n")
    
    
    
  def validate_heuristic(self, prompt, depth_limit=20, temperature=0.8, disable_log='False',lm_plan_file: str = 'lm_plan.tmp', domain_file: str = "data/blocksworld/generated_domain.pddl", config_file: str = "data/blocksworld/bw_config.yaml", **kwargs):

    best_index = -1
    for i, heuristic_obj in enumerate(self.heuristic_functions[self.generation]):
      prompt['next_actions_holding'] = get_next_actions_holding(prompt)
      prompt['next_actions_empty'] = get_next_actions_empty(prompt)
      print(f"GEN: {self.generation}, Heuristic Number: {i},  HEURISTIC:  {str(heuristic_obj['function_text'])} ", flush=True)
      fitness_score = autohd_search(self.base_model, prompt, 
                                    disable_log=disable_log, data_path=self.val_file_path, config_file=config_file, domain_file=domain_file, lm_plan_file=lm_plan_file, step_into_state=True, action_prompt=False,
                                    depth_limit=depth_limit, temperature=temperature, 
                                    heuristic_fn=heuristic_obj['callable_heuristic'], **kwargs)
      
      # fitness_score =random.uniform(0.0, 1.0)
      print(f'GEN: {self.generation}, Heuristic Number: {i} - Fitness Score: {fitness_score}')
      
      heapq.heappush(self.best_heuristics[self.generation], (fitness_score, i, heuristic_obj))
      if fitness_score == 1.0:
        print('Eureka!!! Goated Heuristic Found!!! Voila!!!')
        print(heuristic_obj['function_text'])
        best_index = i
      # Can add stuff to check for errors if any.
    for sample in self.best_heuristics[self.generation][:5]:
      _, _, heuristic_obj = sample
      print(heuristic_obj['function_text'])
    return best_index
  
  def evolve(self, prompt, depth_limit=20, temperature=0.8, disable_log='False', **kwargs):
    sampled_heuristics = None
    
    for gen in range(self.max_generations):
      with open(self.log_dir, 'a') as file:
        file.write(f"Generation {gen}: \n")
      self.best_heuristics[self.generation] = []
      print('Starting generation - ', gen)
      self.get_initial_heuristics(sampled_heuristics=sampled_heuristics)
      _ = self.validate_heuristic(prompt, depth_limit=depth_limit, temperature=temperature, disable_log=disable_log, **kwargs)
      pruned_heuristics = self.prune()
      if len(pruned_heuristics) > self.sample_k:
        sampled_heuristics = self.sample_heuristic(pruned_heuristics=pruned_heuristics)
      else:
        sampled_heuristics = pruned_heuristics
      
      # Log info
      with open(self.log_dir, 'a') as file:
        for idx in range(len(self.best_heuristics[self.generation])):
          acc, _, heuristic_obj = self.best_heuristics[self.generation][idx]
          file.write(f"Function: \n")
          file.write(heuristic_obj['function_text'])
          file.write(f"\n")
          file.write(f"Accuracy: {acc} \n")
    
      # Setup for next generation.
      self.generation += 1
      self.heuristic_functions[self.generation] = []
    return None
  
  # Similar to Fun Search and Citadel, prune the bottom half
  def prune(self):
    return self.best_heuristics[self.generation][len(self.heuristic_functions[self.generation])//2:] # Prune the previous half. 

  def sample_heuristic(self, pruned_heuristics):
    total_items = len(pruned_heuristics) 
    probabilities = [(1/(rank + total_items)) for rank in range(total_items)]
    
    # Normalize the probabilities
    total_probability = sum(probabilities)
    normalized_probabilities = [p / total_probability for p in probabilities]
    
    # Sampled Heuristics
    sampled_heuristics = random.choices(pruned_heuristics, weights=normalized_probabilities, k=self.sample_k)
    return sampled_heuristics
def main(base_lm:Literal['openai','llamaapi'] = 'openai',
         data_path="data/blocksworld/AutoHD_val_set.json",
         test_file_path = "data/blocksworld/split_v1/split_v1_step_2_data.json",
         prompt_path='prompts/blocksworld/pool_prompt_v1.json',   
         temperature=0.8,
         disable_log='False',
         depth_limit=20,
         sc_num=1,
         log_dir="logs/blocksworld/bw_HeuristicSearch_{model_name}.log",
         api_model_id='meta-llama/Meta-Llama-3.1-70B-Instruct',
         openai_model_id='gpt-4o-mini',
         max_generations = 5,
         config_file: str = "data/blocksworld/bw_config.yaml",
         domain_file: str = "data/blocksworld/generated_domain.pddl",
         **kwargs):
  
  with open(prompt_path) as f:
      prompt = json.load(f) 
  
  if base_lm == "openai":
    base_model = OpenAIModel(openai_model_id, additional_prompt="CONTINUE")
    model_name = f'{base_lm}_{openai_model_id}'
  elif base_lm == 'llamaapi':
    base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None, additional_prompt = "CONTINUE")
    model_name = f'{base_lm}_{api_model_id.replace("/", "-")}'
    
  log_dir = log_dir.format(model_name=model_name)  
  heauristic_search = HeuristicSearch(base_model, val_file_path=data_path, log_dir = log_dir, test_file_path=test_file_path, max_generations = max_generations)
  heauristic_search.evolve(prompt=prompt, depth_limit=depth_limit, temperature=temperature, disable_log=disable_log, domain_file=domain_file, config_file=config_file, **kwargs)
  heauristic_search.test_heuristic(prompt=prompt, depth_limit=depth_limit, temperature=temperature, disable_log=disable_log, domain_file=domain_file, config_file=config_file, **kwargs)



if __name__ == '__main__':
    fire.Fire(main)
    