from trl import GRPOConfig, GRPOTrainer
import torch
import numpy as np
from functools import partial
from accelerate import Accelerator
import logging
import math 
from typing import Union

log = logging.getLogger(__name__)
accelerator = Accelerator()
def log_on_main(text):
    if accelerator.is_main_process:
        log.info(text)


class TaskSampler(torch.utils.data.Sampler):
    def __init__(self, 
                 dataset, 
                 num_tasks, 
                 total_iterations, 
                 data_schedule, 
                 batch_size, 
                 mini_repeat_count, 
                 repeat_count, 
                 scheduler_params, 
                 seed=0, 
                 trainer=None):
        """
        Args:
          dataset: a HF dataset; each sample is assumed to be a dict including "task" (an integer 0 to num_tasks-1)
          num_tasks: total number of task categories (e.g. 4)
          total_iterations: total training iterations (T)
          current_iter_fn: callable that returns current iteration (t)
          trainer: reference to the trainer for VREx logging
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.mini_repeat_count = mini_repeat_count
        self.repeat_count = repeat_count
        self.max_dataset_len = len(self.dataset)
        self.num_tasks = num_tasks
        self.total_iterations = total_iterations
        self.data_schedule = data_schedule
        self.trainer = trainer
        self.rng = np.random.default_rng(int(seed))
        task_col = np.asarray(self.dataset['level'])
        self.indices_by_task = {
            t: self.rng.permutation(np.where(task_col == t)[0])
            for t in range(self.num_tasks)
        }
        self.schedule_funcs = {
            'balanced': self._balanced_schedule,
            'cosine': self._cosine_schedule,
            'gaussian': partial(self._gaussian_schedule, **scheduler_params),
            'classic': self._step_schedule,
        }
        log_on_main(f"Data Schedule: {data_schedule}")
        self.schedule_func = self.schedule_funcs[data_schedule]
    
    # Classical Curriculum Learning
    @staticmethod    
    def _step_schedule(t, T, num_tasks):
        active_task = min(int(t * num_tasks / T), num_tasks - 1)
        return dict(enumerate(np.eye(num_tasks)[active_task].tolist()))

    def __iter__(self):
        task_ptrs = {t: 0 for t in range(self.num_tasks)}
        indices_by_task = {t: idx.copy() for t, idx in self.indices_by_task.items()}
        # Don't reset variance regularized state here - it's done once in trainer init
        
        for i in range(self.total_iterations):
            probs_dict = self.schedule_func(i, self.total_iterations, self.num_tasks)
            probs = np.array([probs_dict[j] for j in range(self.num_tasks)])
            # Sample a task for each slot in the batch using the probabilities.
            chosen_tasks = self.rng.choice(np.arange(self.num_tasks), size=self.batch_size, p=probs, replace=True)
            # if rank == 0:
            batch_indices = []

            for t in chosen_tasks:
                bucket = self.indices_by_task[int(t)]
                p = task_ptrs[int(t)] % len(bucket)  # wrap
                batch_indices.append(int(bucket[p]))
                task_ptrs[int(t)] += 1
            
            
            self.last_batch_indices = batch_indices
            self.last_chosen_tasks = chosen_tasks
            
            for _ in range(self.repeat_count):
                for index in batch_indices:
                    for _ in range(self.mini_repeat_count):
                        yield index
            # yield from batch_indices

    def __len__(self):
        return self.total_iterations * self.batch_size * self.mini_repeat_count * self.repeat_count
    @staticmethod
    def _balanced_schedule(t, T, num_tasks):
        return {i: 1. / num_tasks for i in range(num_tasks)}

    @staticmethod
    def _cosine_schedule(t, T, num_tasks):
        total = num_tasks * (num_tasks + 1) / 2.0
        early = {i: (num_tasks - i) / total for i in range(num_tasks)}
        late = {i: (i + 1) / total for i in range(num_tasks)}
        alpha = 0.5 * (1 + math.cos(math.pi * t / T))
        probs = {i: alpha * early[i] + (1 - alpha) * late[i] for i in range(num_tasks)}
        # Enforce symmetric floor equal to the minimum probability in early/late.
        p_min = 2 / (num_tasks * (num_tasks + 1))
        for i in range(num_tasks):
            probs[i] = max(probs[i], p_min)
        norm = sum(probs.values())
        return {i: probs[i] / norm for i in probs}

    @staticmethod
    def _gaussian_schedule(t, T, num_tasks, mu_exp, sigma, min_prob: Union[bool, float]=False, **kwargs):
        '''
        Gaussian schedule for task sampling.
        mu_exp: exponent for the mean, typically 1.0. Move faster at the beginning: < 1.0. Move slower at the beginning: > 1.0
        sigma: standard deviation of the Gaussian distribution
        min_prob: minimum probability for each task
        '''
        # Move mean from 0 to (num_tasks-1) as time progresses, Use sqrt(t / T) to boost the the speed at the beginning
        mu = (t / T) ** mu_exp * (num_tasks - 1)
        p_min = (2 / (num_tasks * (num_tasks + 1))) if (min_prob is True) else (min_prob if isinstance(min_prob, float) else None)
        if p_min is None: raise ValueError("min_prob should be either a boolean or a float")
        if num_tasks * p_min > 1: raise ValueError("num_tasks * p_min must not exceed 1")
        
        # Compute normalized Gaussian probabilities.
        base = [math.exp(-((i - mu) ** 2) / (2 * sigma ** 2)) for i in range(num_tasks)]
        total = sum(base)
        q = [b / total for b in base]
        
        # Mix with uniform floor to guarantee each probability is at least p_min.
        return {i: p_min + (1 - num_tasks * p_min) * q_i for i, q_i in enumerate(q)}


class CurriculumGRPOTrainer(GRPOTrainer):
    def __init__(self, 
                 num_tasks=4, 
                 total_iterations=1200, 
                 data_schedule='balanced', 
                 scheduler_params: dict=None, *args, **kwargs):
        self.num_tasks = num_tasks
        self.total_iterations = total_iterations
        self.data_schedule = data_schedule
        self.scheduler_params=scheduler_params
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self):
        # The parent class passes the dataset as an argument, but we use self.train_dataset

        # generation_batch_size = self.accelerator.num_processes (num_device) * self.args.per_device_train_batch_size (including num_generation)
        return TaskSampler(self.train_dataset,
                           num_tasks=self.num_tasks,
                           total_iterations=self.total_iterations,
                           data_schedule=self.data_schedule,
                           scheduler_params=self.scheduler_params,
                           batch_size= self.args.generation_batch_size * self.args.gradient_accumulation_steps // self.num_generations,
                           mini_repeat_count=self.num_generations,
                           repeat_count=self.num_iterations, # * self.args.steps_per_generation, #num_iterations=1 is a GRPO param.
                           trainer=self)

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Extract task IDs from the batch before processing
        if 'task' in inputs[0]:
            self._current_batch_task_ids = [inp['task'] for inp in inputs]
        
        # Call parent training step
        result = super().training_step(model, inputs, num_items_in_batch)
        
        return result