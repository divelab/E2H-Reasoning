import datasets
import json
from tqdm import tqdm
import torch
import os, pickle
from datetime import datetime
import sys
import random
from reasoners import Evaluator
import copy
# from reasoners.benchmark.bw_augmentor import generate_augmentations
import reasoners.benchmark.bw_utils as bw_utils

def rap_bw_extractor(algo_output):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # to make sure the plan is saved before evaluation in multi-process setting
    try:
        if algo_output.trace is None:
            print("No plan found")
            return ""
        else:
            return "\n".join(algo_output.trace[1])
    except Exception as e:
        print("Error in output extraction,", e)
        return ""

def get_icl(init_prompt, examples):
    icl = init_prompt["intro"] + \
        "\n".join([
            "[STATEMENT]\nAs initial conditions I have that, " + \
            example["init"] + \
            ".\nMy goal is to have that " +\
            example["goal"] + \
            ".\n\nMy plan is as follows:\n\n[PLAN]" + \
            example["plan"]
            for example in examples
        ])
    icl += "\n[STATEMENT]\nAs initial conditions I have that, <init_state>\nMy goal is to <goals>\n\nMy plan is as follows:\n\n[PLAN]\n<action>"
    return icl

def get_take_a_step_back_icl(init_prompt, examples):
    icl = init_prompt["intro"] + \
        "\n".join([
            "[STATEMENT]\nAs initial conditions I have that, " + \
            example["init"] + \
            ".\nMy goal is to have that " +\
            example["goal"] + \
            ".\n\n[EXPLAINATION] Let's apply logic step by step to propose the plan." +\
            example["explaination"] + \
            ".\n\nMy plan is as follows:\n\n[PLAN]" + \
            example["plan"]
            for example in examples
        ])
    icl += "\n[STATEMENT]\nAs initial conditions I have that, <init_state>\nMy goal is to <goals>\n\nMy plan is as follows:\n\n[PLAN]\n<action>"
    return icl

class BWEvaluator(Evaluator):
    def __init__(self, 
                 config_file,
                 domain_file,
                 data_path,
                 init_prompt,
                 disable_log=False,
                 disable_tqdm=False,
                 output_extractor=rap_bw_extractor,
                 answer_extractor=lambda x:x,
                 sample_prompt_type="rap",
                 mode="majority"
                 ) -> None:
        super().__init__()
        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.input_processor = lambda x: x
        if 'step_1' in data_path:
            from blocksworld_reward_model import BlocksWorldModel
            self.is_step1 = True
            self.evaluator = BlocksWorldModel
            question = "\n[STATEMENT]\nAs initial conditions I have that, the orange block is clear, the hand is empty, the blue block is on top of the red block, the orange block is on top of the blue block and the red block is on the table.\n\nMy plan is as follows:\n\n[PLAN]\n"
            with open(data_path, 'r') as f:
                data = json.load(f)
            self.full_dataset = []
            for i, d in enumerate(data):
                question_obj = {}
                question_obj['init'] = d['init']
                question_obj['goal'] = d['goal']
                question_obj['plan'] = d['plan']
                question_obj['question'] = question
                self.full_dataset.append(question_obj)
        else:
            self.is_step1 = False        
            self.full_dataset = bw_utils.load_blocksworld(config_file, domain_file, data_path)  # [{"goal": str, "init": str}]
        self._dataset_name = 'blocksworld'
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.sample_prompt_type = sample_prompt_type
        self.mode = mode
        self.lm_plan_file = "tmp_plan.txt"
        self.config_file = config_file
        self.domain_file = domain_file
        print(self.mode)

    def sample_prompt(self,
                      shuffle_prompt=True,
                      num_shot=4):

        sample_prompt_type = self.sample_prompt_type
      
        if sample_prompt_type == "rap":
            if shuffle_prompt:
                examples = random.sample(self.init_prompt["example_pool"], num_shot)
            else:
                examples = self.init_prompt["example_pool"][:num_shot]

            icl = get_icl(self.init_prompt, examples)
            
            prompt = copy.deepcopy(self.init_prompt)
            prompt["icl"] = icl
            prompt["icl_list"] = [icl]
            examples = copy.deepcopy(examples)
            for i in range(5):
                new_examples = []
                for example in examples:
                    if len(example["states"]) > 1:
                        new_examples.append({
                            "init": example["states"][0],
                            "goal": example["goal"],
                            "plan": "\n" + "\n".join(example["plan"].split("\n")[3:]),
                            "states": example["states"][1:]
                        })
                    else:
                        new_examples.append(example)
                examples = copy.deepcopy(new_examples)
                # print("EXAMPLES: ",examples,flush=True)
                icl = get_icl(self.init_prompt, examples)
                prompt["icl_list"].append(icl)
        elif sample_prompt_type == "o1":
            prompt = {}
            prompt['o1'] = self.init_prompt["intro"]
            with open('prompts/blocksworld/o1_prompt.json') as f:
              init_prompt = json.load(f)
              prompt['o1'] += init_prompt["o1_prompt"]
            prompt['o1'] += "\n[STATEMENT]\nAs initial conditions I have that, <init_state>\nMy goal is to <goals>\n\nMy plan is as follows:\n\n[PLAN]\n<action>"
        else:
            raise NotImplementedError
        print()
        print()
        print()
        # print("prompt:",  prompt)
        # print("------------------------------------------------------------------")
        return prompt
    
    def sample_automatic_prompt(self,
                      shuffle_prompt=True,
                      num_shot=4):   # For APE

        sample_prompt_type = self.sample_prompt_type
      
        if sample_prompt_type == "rap":
            examples = self.init_prompt["example_pool"]
            assert len(examples)==10 # For BW

            # icl = get_icl(self.init_prompt, examples)
            
            # prompt = copy.deepcopy(self.init_prompt)
            # prompt["icl"] = icl
            # prompt["icl_list"] = [icl]
            examples = copy.deepcopy(examples)
            print("EXAMPLES1 : ",examples,flush=True)
            print("EXAMPLES1 : ",len(examples),flush=True)

            init=[]
            goal=[]
            plan=[]
            for ex in examples:
                init.append(ex['states'][0])
                goal.append(ex['goal'])
                plan.append("\n".join(ex["plan"].split("\n")[3:]))
            
        else:
            raise NotImplementedError
        # print()
        # print()
        # print()
        # print("prompt:",  prompt)
        # print("------------------------------------------------------------------")
        dataset = [{'init': init[i], 'goal': goal[i], 'plan': plan[i]} for i in range(len(init))]

        return dataset
    def eval_output(self, answer, outputs):
        if isinstance(outputs, str):
            outputs = [outputs]
        # else assume it's already a list of strings

        def check_one(output):
            if self.is_step1:
                evaluator = self.evaluator(answer['init'], answer['goal'], answer['plan'])
                try:
                    final_state = evaluator.simulate_plan(output)
                    return evaluator.check_goal(final_state)[0]
                except Exception:
                    return False
            else:
                bw_utils.text_to_plan_blocksworld(
                    output,
                    answer["instance_file"],
                    self.config_file,
                    self.domain_file,
                    self.lm_plan_file
                )
                return bw_utils.validate_plan(
                    self.domain_file,
                    answer["instance_file"],
                    self.lm_plan_file
                )[0]

        if self.mode == 'pass':
                # Return a list of booleans, one per candidate
            return any(check_one(o) for o in outputs)

        elif self.mode == 'majority':
                # Only one collapsed output expected—evaluate and return its correctness
            return bool(check_one(outputs[0]))

        else:
            raise ValueError(f"Unknown eval mode: {self.mode}")
    
if __name__ == "__main__":
    config_file: str = "data/blocksworld/bw_config.yaml"
    steps = 1
    domain_file: str = "data/blocksworld/generated_domain.pddl"
    data_path=f'data/blocksworld/split_v1/split_v1_step_{steps}_data.json'
    prompt_path='prompts/blocksworld/pool_prompt_v1.json'
    
    with open(prompt_path) as f:
        prompt = json.load(f)
    
    def sc_output_extractor(algo_output):
        from collections import Counter
        answers = [x for x in algo_output if x is not None]
        counter = Counter(answers)
        if counter == {}:
            return None
        return counter.most_common(1)[0][0]
    
    evaluator = BWEvaluator(config_file=config_file, 
                            domain_file=domain_file, 
                            data_path=data_path, 
                            init_prompt=prompt, 
                            disable_log=False, 
                            output_extractor=sc_output_extractor, 
                            sample_prompt_type="rap")
    dataset =evaluator.full_dataset
    print("Dataset length: ", len(dataset))
    train_set = []
    for i in range(len(dataset)):
        init = dataset[i]['init']
        goal = dataset[i]['goal']
        plan = dataset[i]['plan']
        instance_file = dataset[i]['instance_file'] 
        aug_data = generate_augmentations(init, goal, plan, num_augmentations=15)
        for key in aug_data.keys():
            for aug in aug_data[key]:
                training_instance = {}
                training_instance['init'] = aug['init']
                training_instance['goal'] = aug['goal']
                training_instance['plan'] = aug['plan']
                training_instance['instance_file'] = instance_file
                training_instance['augmentation_type'] = key
                training_instance['mapping'] = aug['mapping']
                train_set.append(training_instance)
    print('Dataset Length', len(train_set))
    with open(f'data/blocksworld/train_set-{steps}-more.json', 'w') as f:
        json.dump(train_set, f, indent=4)
    