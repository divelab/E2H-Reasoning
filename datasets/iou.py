import os
import numpy as np
from pprint import pprint
from datasets import load_dataset


dataset_name = 'gsm8k'
split = 'test'
results = dict()
for task in ['trivial', 'easy', 'medium', 'hard']:
    old = load_dataset('json', data_dir=os.path.join('datasets', dataset_name+'_old', task), split=split)
    new = load_dataset('json', data_dir=os.path.join('datasets', dataset_name, task), split=split)

    prompts = np.array([*old['question'], *new['question']])
    unique, counts = np.unique(prompts, return_counts=True)
    
    iou = (counts > 1).sum() / len(unique)
    results[task] = iou
pprint(results, indent=4, width=2)