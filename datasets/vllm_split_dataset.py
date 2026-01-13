import os
import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def preprocess_aqua(dataset):
    dataset = dataset.rename_columns({
        'rationale' : 'solution',
        'correct' : 'answer'
    })
    return dataset
def chat_template_aqua(**kwargs):
    question = kwargs['question']
    options = "  ".join(kwargs['options'])
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
        },
        {
            "role": "user",
            "content": f"Solve the following math problem and choose an answer from the given options\n{question}\n{options}\n\n Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> C </answer>."
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        }
    ]
    return messages
def is_correct_aqua(completion, answer):
    answer_match = re.findall(r'<answer>\s*(.*?)\s*</answer>', completion, re.DOTALL)
    if len(answer_match) > 0:
        if answer_match[-1].strip() == answer:
            return 1
    return 0


def preprocess_gsm8k(dataset):
    def preprocess(example):
        answer = example['answer'].replace(",", "")
        answer = answer[answer.find('####') + 4:].strip()
        answer = answer.split()[0]
        return {
            'question' : example['question'],
            'solution' : example['answer'],
            'answer' : answer
        }
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    return dataset
def chat_template_gsm8k(**kwargs):
    question = kwargs['question']
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
        },
        {
            "role": "user",
            "content": f"Solve the following math problem\n{question}\n\n Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 500 </answer>."
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        }
    ]
    return messages
def is_correct_gsm8k(completion, answer):
    answer = float(answer)
    answer_match = re.findall(r'<answer>\s*(.*?)\s*</answer>', completion, re.DOTALL)
    if len(answer_match) > 0:
        extracted_answer = answer_match[-1].strip()
        extracted_answer = extracted_answer.replace(",", "")
        extracted_answer = re.search(r'-?\d+(?:\.\d+)?', extracted_answer)
        if extracted_answer is not None:
            extracted_answer = float(extracted_answer.group(0))
            return 1 * (abs(answer - extracted_answer) < 1e-5)
    return 0


def split_into_tasks(dataset, num_splits):
    quantiles = [
        np.quantile(dataset['difficulty'], i/num_splits) 
        for i in range(1, num_splits)
    ] + [np.max(dataset['difficulty'])]

    tasks = dict((f'task{i}', []) for i in range(1, num_splits+1))
    for example_idx in range(len(dataset)):
        for task_idx, quantile in enumerate(quantiles):
            if dataset['difficulty'][example_idx] <= quantile:
                tasks[f'task{task_idx+1}'].append(example_idx)
                break

    for task in tasks:
        tasks[task] = dataset.select(tasks[task])
    
    return tasks


def main(config):

    tokenizer = AutoTokenizer.from_pretrained(
        config['model_params']['model'], 
        trust_remote_code=config['model_params']['trust_remote_code']
    )
    model = LLM(
        **config['model_params'], 
        seed=config['seed'],
        task='generate',
        max_model_len=config['sampling_params']['truncate_prompt_tokens']+config['sampling_params']['max_tokens']
    )
    sampling_params = SamplingParams(
        **config['sampling_params'], 
        seed=config['seed']
    )

    for split in ['test', 'train']:
        print('\n\n*****')
        print(split)
        print('*****')

        dataset = load_dataset(**config['dataset'], split=config[split]['split'])

        if config[split]['size'] != -1:
            dataset = dataset.shuffle(seed=config['seed'])
            dataset = dataset.select(range(config[split]['size']))

        dataset = config['dataset_preprocess_func'](dataset)
        assert set(['question', 'solution', 'answer']).issubset(dataset.column_names), 'Every dataset needs to have question, solution, answer columns'

        prompts = [
            tokenizer.apply_chat_template(
                config['chat_template_func'](**example), 
                tokenize=False, 
                continue_final_message=True
            )
            for example in dataset
        ]

        outputs = model.generate(prompts, sampling_params)

        is_correct = np.array([
            [
                config['correctness_func'](completion_output.text, dataset['answer'][request_idx]) 
                for completion_output in request_output.outputs
            ]
            for request_idx, request_output in tqdm(enumerate(outputs), desc='Checking Correctness')
        ])
        
        difficulty = sampling_params.n - is_correct.sum(axis=1)
        dataset = dataset.add_column('difficulty', difficulty.tolist())

        if config['num_splits'] == 1:
            dataset.to_json(os.path.join(config['save_path'], split+'.jsonl'))
        else:
            for task_name, dataset in split_into_tasks(dataset, config['num_splits']).items():
                dataset.to_json(os.path.join(config['save_path'], task_name, split+'.jsonl'))

    
if __name__ == '__main__':

    config = {

        'dataset' : {
            'path' : 'openai/gsm8k',
            'name' : 'main'
        },
        
        'save_path' : 'datasets/gsm8k',

        'seed' : 42,

        'train' : {
            'split' : 'train',
            'size' : -1
        },
        'test' : {
            'split' : 'test',
            'size' : -1
        },

        'dataset_preprocess_func' : preprocess_gsm8k,

        'chat_template_func' : chat_template_gsm8k,

        'model_params' : {
            'model' : 'Qwen/Qwen2.5-1.5B-Instruct',
            'trust_remote_code' : True,
            'tensor_parallel_size' : 2,  # num gpus 2/4/8
            'dtype' : 'bfloat16',
            'gpu_memory_utilization' : 0.9
        },

        'sampling_params' : {
            'n' : 20,  # max difficulty level
            'temperature' : 0.8,
            'max_tokens' : 512,
            'min_tokens' : 1,
            'truncate_prompt_tokens' : 1600
        },

        'correctness_func' : is_correct_gsm8k,

        'num_splits' : 4,

    }
    main(config)
