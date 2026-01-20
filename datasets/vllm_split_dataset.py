import os
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import (
    load_dataset, 
    concatenate_datasets, 
    get_dataset_config_names
)
import math_utils


def preprocess_aqua(dataset):
    def preprocess(example):
        answer = example['correct'].strip()
        if answer is None:
            return
        else:
            return {
                'question' : example['question'] + '\n' + '  '.join(example['options']),
                'solution' : example['rationale'],
                'answer' : answer
            }
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    return dataset
def chat_template_aqua(question, solution, answer, dataset_name):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
        },
        {
            "role": "user",
            "content": f"Solve the following math problem\n{question}\n\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> C </answer>."
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        }
    ]
    return messages


def preprocess_gsm8k(dataset):
    def preprocess(example):
        answer = example['answer'].split('####')[-1].strip()
        answer = answer.replace(",", "")
        if answer is None:
            return
        else:
            return {
                'question' : example['question'],
                'solution' : example['answer'],
                'answer' : str(int(answer))
            }
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    return dataset
def chat_template_gsm8k(question, solution, answer, dataset_name):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
        },
        {
            "role": "user",
            "content": f"Solve the following math problem\n{question}\n\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 500 </answer>."
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        }
    ]
    return messages


def preprocess_math(dataset):
    def preprocess(example):
        answer = math_utils.remove_boxed(math_utils.last_boxed_only_string(example['solution']))
        if answer is None:
            return
        else:
            return {
                'question' : example['problem'],
                'solution' : example['solution'],
                'answer' : answer
            }
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    return dataset
def chat_template_math(question, solution, answer, dataset_name):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
        },
        {
            "role": "user",
            "content": f"Solve the following math problem\n{question}\n\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 500 </answer>."
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        }
    ]
    return messages


def load_and_prepare_dataset(dataset_name, dataset_cfg, split, seed):

    if dataset_cfg['subset'] is not None:
        data = load_dataset(
            path=dataset_cfg['path'],
            name=dataset_cfg['subset'],
            split=split
        )
    else:
        data = concatenate_datasets([
            load_dataset(
                path=dataset_cfg['path'],
                name=subset,
                split=split
            )
            for subset in get_dataset_config_names(dataset_cfg['path'])
        ])

    if dataset_cfg[split]['size'] != -1:
        data = data.shuffle(seed=seed)
        data = data.select(range(dataset_cfg[split]['size']))

    data = dataset_cfg['dataset_preprocess_func'](data)

    required_columns = {'question', 'solution', 'answer'}
    assert required_columns.issubset(data.column_names)

    data = data.add_column('dataset_name', [dataset_name] * len(data))

    return data


def split_into_tasks(dataset, num_splits):
    task_ids = np.digitize(
        dataset["difficulty"], 
        np.quantile(
            dataset["difficulty"],
            np.linspace(1 / num_splits, 1, num_splits)
        ), 
        right=True
    )
    tasks = {}
    for task_idx in range(num_splits):
        indices = np.where(task_ids == task_idx)[0]
        tasks[f"task{task_idx + 1}"] = dataset.select(indices)
    return tasks


def main(config):

    model = LLM(
        **config['model_params'], 
        seed=config['seed'],
        task='generate',
        max_model_len=config['sampling_params']['truncate_prompt_tokens']+config['sampling_params']['max_tokens']
    )
    tokenizer = model.get_tokenizer()
    sampling_params = SamplingParams(
        **config['sampling_params'], 
        seed=config['seed']
    )

    for split in ['test', 'train']:
        print('\n\n*****')
        print(split)
        print('*****')

        dataset = concatenate_datasets([
            load_and_prepare_dataset(dataset_name, dataset_cfg, split, config['seed'])
            for dataset_name, dataset_cfg in config['datasets'].items()
        ])
        dataset = dataset.shuffle(seed=config['seed'])
        print("Dataset Size: ", len(dataset))

        prompts = [
            tokenizer.apply_chat_template(
                config['datasets'][example['dataset_name']]['chat_template_func'](**example), 
                tokenize=False, 
                continue_final_message=True
            )
            for example in dataset
        ]

        outputs = model.generate(prompts, sampling_params)

        is_correct = np.array([
            [
                math_utils.is_equiv(completion_output.text, dataset['answer'][request_idx]) 
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

        'datasets' : {

            'gsm8k' : {
                'path' : 'openai/gsm8k',
                'subset' : 'main',
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
            },

            'aqua' : {
                'path' : 'deepmind/aqua_rat',
                'subset' : 'raw',
                'train' : {
                    'split' : 'train',
                    'size' : 6000
                },
                'test' : {
                    'split' : 'test',
                    'size' : -1
                },
                'dataset_preprocess_func' : preprocess_aqua,
                'chat_template_func' : chat_template_aqua,
            },

            'math' : {
                'path' : 'EleutherAI/hendrycks_math',
                'subset' : None,
                'train' : {
                    'split' : 'train',
                    'size' : -1
                },
                'test' : {
                    'split' : 'test',
                    'size' : -1
                },
                'dataset_preprocess_func' : preprocess_math,
                'chat_template_func' : chat_template_math,
            }

        },

        'save_path' : 'arithmetic',

        'seed' : 42,

        'model_params' : {
            'model' : 'Qwen/Qwen2.5-1.5B-Instruct',
            'trust_remote_code' : True,
            'tensor_parallel_size' : 2,
            'data_parallel_size' : 1,
            'dtype' : 'bfloat16',
            'gpu_memory_utilization' : 0.95
        },

        'sampling_params' : {
            'n' : 32,  # max difficulty level
            'temperature' : 0.8,
            'max_tokens' : 1024,
            'min_tokens' : 1,
            'truncate_prompt_tokens' : 2048
        },

        'num_splits' : 4,

    }
    main(config)
