import torch
import numpy as np
from typing import Union
from trl import GRPOTrainer
from functools import partial


class TaskSampler(torch.utils.data.Sampler):
    def __init__(
        self, 
        data_source, 
        mini_repeat_count, 
        batch_size, 
        repeat_count, 
        seed,
        total_iterations, 
        scheduler_params,
        resample_size 
    ):
        self.dataset = data_source
        assert 'level' in self.dataset.column_names
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.rng = np.random.default_rng(int(seed))
        self.total_iterations = total_iterations
        self.schedule_func = {
            'balanced': self._balanced_schedule,
            'cosine': self._cosine_schedule,
            'gaussian': partial(self._gaussian_schedule, **scheduler_params['scheduler_args']),
            'classic': self._step_schedule,
        }[scheduler_params['curriculum_schedule']]
        self.resample_size = resample_size

        self.num_tasks = len(set(self.dataset['level']))
        task_col = np.asarray(self.dataset['level'])
        self.indices_by_task = {
            t: self.rng.permutation(np.where(task_col == t)[0])
            for t in range(self.num_tasks)
        }
        self.max_dataset_len = len(self.dataset)

    # Classical Curriculum Learning
    @staticmethod    
    def _step_schedule(t, T, num_tasks):
        active_task = min(int(t * num_tasks / T), num_tasks - 1)
        return dict(enumerate(np.eye(num_tasks)[active_task].tolist()))

    @staticmethod
    def _balanced_schedule(t, T, num_tasks):
        return {i: 1. / num_tasks for i in range(num_tasks)}

    @staticmethod
    def _cosine_schedule(t, T, num_tasks):
        total = num_tasks * (num_tasks + 1) / 2.0
        early = {i: (num_tasks - i) / total for i in range(num_tasks)}
        late = {i: (i + 1) / total for i in range(num_tasks)}
        alpha = 0.5 * (1 + np.cos(np.pi * t / T))
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
        base = [np.exp(-((i - mu) ** 2) / (2 * sigma ** 2)) for i in range(num_tasks)]
        total = sum(base)
        q = [b / total for b in base]
        
        # Mix with uniform floor to guarantee each probability is at least p_min.
        return {i: p_min + (1 - num_tasks * p_min) * q_i for i, q_i in enumerate(q)}

    def __len__(self):
        return self.total_iterations * self.batch_size * self.mini_repeat_count * self.repeat_count

    def __iter__(self):
        task_ptrs = {t: 0 for t in range(self.num_tasks)}
        indices_by_task = {t: idx.copy() for t, idx in self.indices_by_task.items()}
        # Don't reset variance regularized state here - it's done once in trainer init
        
        for i in range(self.total_iterations):
            self.current_iteration = i
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


class CurriculumGRPOTrainer(GRPOTrainer):
    def __init__(
        self, 
        scheduler_params, 
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.scheduler_params = scheduler_params

    def _get_train_sampler(self, dataset=None):
        if dataset is None:
            dataset = self.train_dataset
        return TaskSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size * self.args.gradient_accumulation_steps // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
            total_iterations=self.args.max_steps,
            scheduler_params=self.scheduler_params,
            resample_size=self.args.generation_batch_size // (self.accelerator.num_processes * self.num_generations)
        )

    def _compute_loss(self, model, inputs):
        dapo_iter = 0

        while dapo_iter < self.scheduler_params.max_dapo_iter:
            grouped_advantages = inputs["advantages"].reshape(-1, self.num_generations)
            if not torch.any(torch.all(torch.abs(grouped_advantages) < 1e-6, dim=-1)):
                break
            
            dapo_iter += 1
            print("Dapo Iter: ", dapo_iter)
            # generation_batch = [self.callback_handler.train_dataloader.dataset[idx] for idx in self.callback_handler.train_dataloader.batch_sampler.batch_sampler.sampler.resample()]
            generation_batch = [self.callback_handler.train_dataloader.dataset[idx] for idx in range(8)]
            inputs = self._prepare_inputs(generation_batch)
        return super()._compute_loss(model, inputs)