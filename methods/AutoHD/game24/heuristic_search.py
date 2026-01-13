import json
from reasoners.lm.openai_model import OpenAIModel
from inference import autohd_search
from typing import Literal
import fire
import heapq
import random
import re
from utils import timeout_handler, TimeoutException, extract_heuristics, create_callable_function
from reasoners.lm.llama_api_model import LLaMaApiModel
import signal
from evolution_prompts import init_prompt, evolution_prompt, types, evolution_type

class HeuristicSearch:
  def __init__(self,
               base_model, 
               temperature=0.8, 
               init_prompt=init_prompt, 
               evolution_prompt=evolution_prompt, 
               val_file_path=None,
               test_file_path=None, 
               max_generations=10, 
               sample_k=3, 
               n_candidate=4, 
               log_dir = None):
    self.base_model = base_model
    self.init_prompt = init_prompt
    self.n_candidate = n_candidate
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
    self.evolution_prompt = evolution_prompt
    self.log_dir = log_dir
    
  def get_initial_heuristics(self, sampled_heuristics=None):

    print("ENTERED GET INITIAL HEURISTICS --- CHECKPOINT!!!")
    while len(self.heuristic_functions[self.generation]) < self.sample_k: # atleast 3 heuristics.
      
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
        
        print("--------================-------------====================== ------------------",flush=True)
        llm_outputs = self.base_model.generate([prompts],
                                  num_return_sequences=self.n_candidate,
                                  #max_length=20,
                                  temperature=self.temperature,
                                  do_sample=True,
                                  hide_input=True,
                                  top_p = 0.95,
                                  ).text
        for idx in range(self.n_candidate):
          parsed_output = self.parse_output(llm_outputs[idx])
          if parsed_output is not None:
              heuristic_function_obj = {'description': parsed_output[0], 'function_text': parsed_output[1], 'llm_output': llm_outputs[idx]}
              self.get_callable_function(idx,heuristic_function_obj)
            
        if self.generation == 0:
          break

    print(f'Initial {len(self.heuristic_functions[self.generation])} Heuristics collected.')

  def get_callable_function(self, id, heuristic_function_obj):
    code = heuristic_function_obj['function_text']
    heuristic_fn = create_callable_function(code=code)
    # Discard wrong heuristic functions, by dummy state.
    try:
        # Set a timeout for the heuristic function call
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # Timeout duration in seconds
      
        heuristic_fn([1,2,3,4])
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
  
  def test_heuristic(self, 
                           prompt, 
                           depth_limit=4, 
                           disable_log='False',
                           lm_plan_file: str = 'lm_plan.tmp',
                           **kwargs):
        
        print("Best heuristic in the last generation: ")
        best_item = max(self.best_heuristics[self.max_generations - 1], key=lambda x: x[0])
        best_heuristic_obj = best_item[2]
        
        accuracy = autohd_search(
          beam_size=5,
          base_model=self.base_model,
          prompt=prompt,
          depth_limit=depth_limit,
          search_algo='beamstar',
          temperature=self.temperature,
          disable_log=disable_log,
          data_path=self.test_file_path,
          n_iters=5,
          heuristic_fn=best_heuristic_obj['callable_heuristic'],
          heuristic_search_type='test',
          **kwargs
        )
        
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
        all_items = [item for sublist in self.best_heuristics.values() for item in sublist]
        # Find the tuple with the highest fitness_score
        best_item = max(all_items, key=lambda x: x[0])
        # Extract the heuristic_obj
        best_heuristic_obj = best_item[2]
        accuracy = autohd_search(
            beam_size=5,
            base_model=self.base_model,
            prompt=prompt,
            depth_limit=depth_limit,
            search_algo='beamstar',
            temperature=self.temperature,
            disable_log=disable_log,
            data_path=self.test_file_path,
            n_iters=5,
            heuristic_fn=best_heuristic_obj['callable_heuristic'],
            heuristic_search_type='test',
            **kwargs
          )
        
        print("Function: ")
        print(best_heuristic_obj['function_text'])
        print(f"Accuracy is {accuracy}")
        
        with open(self.log_dir, 'a') as file:
            file.write(f"Test the best heuristic in all the generations: \n")
            file.write(f"Function: \n")
            file.write(best_heuristic_obj['function_text'])
            file.write(f"\n")
            file.write(f"Accuracy: {accuracy} \n") 
  
   
  def validate_heuristic(self, 
                         prompt, 
                         depth_limit=4, 
                         disable_log='False',
                        **kwargs):

    best_index = -1
    for i, heuristic_obj in enumerate(self.heuristic_functions[self.generation]):
      print(f"GEN: {self.generation}, Heuristic Number: {i},  HEURISTIC:  {str(heuristic_obj['function_text'])} ", flush=True)
      fitness_score = autohd_search(
          beam_size=5,
          base_model=self.base_model,
          prompt=prompt,
          depth_limit=depth_limit,
          search_algo='beamstar',
          temperature=self.temperature,
          disable_log=disable_log,
          data_path=self.val_file_path,
          heuristic_fn=heuristic_obj['callable_heuristic'],
          n_iters=5,
          terminal_beam=1,
          **kwargs
      )
      
      print(f'GEN: {self.generation}, Heuristic Number: {i} - Fitness Score: {fitness_score}')
      
      heapq.heappush(self.best_heuristics[self.generation], (fitness_score, i, heuristic_obj))
      if fitness_score == 1.0:
        print('Eureka!!! Goated Heuristic Found!!! Voila!!!')
        print(heuristic_obj['function_text'])
        best_index = i
      # Can add stuff to check for errors if any.
    for sample in self.best_heuristics[self.generation][:5]:
      print(f"Top 5 heuristics in generation {self.generation}")
      _, _, heuristic_obj = sample
      print(heuristic_obj['function_text'])
    return best_index
  
  # Preload the heuristics from the log file of previous experiments/evolutions.
  def preload_heuristics(self, heuristic_functions):
    for i, heuristic_obj in enumerate(heuristic_functions):
      self.heuristic_functions[i] = []
      self.best_heuristics[i] = []
      heuristicfunc_obj = {'function_text': heuristic_obj['function'], 'callable_heuristic': None}
      heuristicfunc_obj['callable_heuristic'] = self.get_callable_function(0, heuristicfunc_obj)
      self.heuristic_functions[i].append((heuristicfunc_obj))
      self.best_heuristics[i].append((heuristic_obj['accuracy'],i,heuristicfunc_obj))
  
  def evolve(self, prompt, depth_limit=4, disable_log='False', **kwargs):
    sampled_heuristics = None
    
    for gen in range(self.max_generations):
      with open(self.log_dir, 'a') as file:
          file.write(f"Generation {gen}: \n")
      self.best_heuristics[self.generation] = []
      print('Starting generation - ', gen)
      self.get_initial_heuristics(sampled_heuristics=sampled_heuristics)
      _ = self.validate_heuristic(prompt, depth_limit=depth_limit, disable_log=disable_log, **kwargs)
      # if best_heuristic_index !=-1:
      #   return self.heuristic_functions[self.generation][best_heuristic_index]
      pruned_heuristics = self.prune()
      # print("DOING CITADEL 2",len(pruned_heuristics), flush= True)
      sampled_heuristics = self.sample_heuristic(pruned_heuristics=pruned_heuristics)
      
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
    # print('No Heuristic Found that solves all.')
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
  
def main(base_lm:Literal['openai', 'api']='openai', 
         val_file_path=None,
         prompt_path='prompts/game24/prompts.json',
         test_file_path='data/game24/val-autohd.csv',  
         temperature=0.8,
         disable_log='False',
         depth_limit=4,
         sc_num=5,
         max_generations=5,
         log_dir="examples/A-star/game24/gam24_HeuristicSearch_10gen-{model_name}.log",
         openai_model_id='gpt-4o-mini',
         api_model_id='meta-llama/Meta-Llama-3.1-70B-Instruct',
         fetch_heuristics_from_log=False,
         **kwargs):
  
  with open(prompt_path) as f:
      prompt = json.load(f) 
  
  if base_lm == 'openai':
    base_model = OpenAIModel(openai_model_id, additional_prompt="CONTINUE")
    model_name = f'{base_lm}_{openai_model_id}'
  elif base_lm == 'api':
    base_model = LLaMaApiModel(None, None, use_api=True, model_id=api_model_id, quantized=None, additional_prompt="CONTINUE")
    model_name = f'{base_lm}_{api_model_id.replace("/", "-")}'
  
  log_dir = log_dir.format(model_name=model_name)
  heauristic_search = HeuristicSearch(base_model, val_file_path=val_file_path, n_candidate=sc_num, max_generations=max_generations, log_dir=log_dir, test_file_path=test_file_path, temperature=temperature)
  if fetch_heuristics_from_log:
    with open(log_dir, 'r') as file:
      content = file.read()
    _, best_per_generation, _ = extract_heuristics(content)
    heauristic_search.preload_heuristics(best_per_generation)
  else:
    heauristic_search.evolve(prompt=prompt, depth_limit=depth_limit, disable_log=disable_log, **kwargs)
  heauristic_search.test_heuristic(prompt=prompt, depth_limit=depth_limit, disable_log=disable_log, **kwargs)



if __name__ == '__main__':
    fire.Fire(main)
    