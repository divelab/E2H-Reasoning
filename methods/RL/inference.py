from reasoners.benchmark import BWEvaluator
from reasoners.lm import HFModel
from datetime import datetime
import json
import fire
from utils import sc_output_extractor, generate_icl


class RLReasoner():
    def __init__(self, base_model, temperature=0.8, sc_num = 1, model_type="completion", icl_example="", pass_at_k=1):
        self.base_model = base_model
        self.temperature = temperature
        self.model_type = model_type
        assert not (sc_num > 1 and pass_at_k > 1), "sc_num > 1 and pass_at_k > 1 is not supported"
        if sc_num > 1:
            self.num_generations = sc_num
        else:
            self.num_generations = pass_at_k
        self.tokenizer = base_model.tokenizer
        self.icl_example = icl_example
    
    def get_r1_prompt(self, example):
        r1_prefix = [{
                "role": "system",
                "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
            },
            { 
                "role": "user",
                "content": f"I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nHere is the format of the actions: \n\npick up the [block_name] block # for example: pick up the blue block\nunstack the [block_name] block from on top of the [another_block_name] block # for example: unstack the orange block from on top of the black block\nput down the [block_name] block # for example put down the red block\nstack the [block_name] block on top of the [another_block_name] block # for example: stack the yellow block on top of the red block \n\n{self.icl_example}\n\nHere is the initial state of the blocks: {example['init']}\n\nHere is the goal state of the blocks: {example['goal']}.\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer>\nunstack the cyan block from on top of the emerald block\nput down the cyan block</answer>\n" # , Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer>\nunstack the cyan block from on top of the emerald block\nput down the cyan block</answer>\n for example: <plan>\npick up the blue block\nstack the blue block on top of the yellow block\nunstack the orange block from on top of the black block\nstack the orange block on top of the red block</plan>
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>"
            }
        ]
        return self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True)
        # return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "plan": plan, "init": init, "goal": goal}
    
    def __call__(self, example, prompt=None):
        # inputs = prompt["icl"].replace("<init_state>", example["init"])\
        #     .replace("<goals>", example["goal"]).replace("<action>", "")
        if isinstance(example, list):
            inputs = [self.get_r1_prompt(ex) for ex in example]
        else:
            inputs = [self.get_r1_prompt(example)]
        outputs = []
        print('Total Number of Generations:', self.num_generations)
        for _ in range(self.num_generations):
          if self.model_type == "completion":   
              outputs.append(self.base_model.generate(inputs,
                                            hide_input=True,
                                            do_sample=True,
                                            skip_special_tokens=False,
                                            temperature=self.temperature).text) 
        outputs = [list(group) for group in zip(*outputs)]
        return outputs    

def main(model_checkpoint=300,
         model_dir='qwen-bw-r1-aha-moment/deep_seek-r1-2step-1.5/checkpoint-{num}',
         steps=2,
         config_file: str = "data/blocksworld/bw_config.yaml", 
         domain_file: str = "data/blocksworld/generated_domain.pddl", 
         resume=0, 
         log_dir=None,
         temperature=0.0,
         prompt_path='prompts/blocksworld/pool_prompt_v1.json',
         sc_num=1,
         use_icl = False,
         use_vllm = False,
         max_batch_size=64,
         pass_at_k=1
         ):
    print('Running BW inference...')
    model_dir = model_dir.format(num=model_checkpoint)
    data_path='data/blocksworld/split_v1/split_v1_step_{steps}_data.json'.format(steps=steps)
    model_name= model_dir.split('/')[-1]
    log_dir =  f'logs/Blocksworld/'\
                        f'RL/step_{steps}/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    log_dir = log_dir + f'_{model_name}_t_{temperature}_sc_{sc_num}'
    
    with open(prompt_path) as f:
        prompt = json.load(f)
    
    with open('data/blocksworld/train_set-2-more_with_trace.json') as f:
        icl_examples = json.load(f)
    icl=""
    if use_icl:
        icl = generate_icl(icl_examples, provide_think_icl=False, num_icl = 2)
    print(icl)
    mode="majority" if pass_at_k == 1 else "pass"
    base_model = HFModel(model_pth=model_dir, tokenizer_pth=model_dir, max_new_tokens=512, max_batch_size=max_batch_size)
    reasoner = RLReasoner(base_model, temperature=temperature, sc_num=sc_num, icl_example=icl, pass_at_k=pass_at_k)
    evaluator = BWEvaluator(config_file=config_file, domain_file=domain_file, data_path=data_path, init_prompt=prompt, disable_log=False, output_extractor=lambda x: sc_output_extractor(x, mode=mode), sample_prompt_type="rap", mode=mode) # rap prompt includes cot
    # accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    accuracy = evaluator.batched_evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir, batch_size=max_batch_size)
    print('Accuracy: ', accuracy)


if __name__ == "__main__":
    fire.Fire(main)    