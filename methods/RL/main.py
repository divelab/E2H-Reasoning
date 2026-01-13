import os
import sys
from typing import Union, TypedDict, List
from typing import Any, Union
from trl.trainer.utils import pad
from contextlib import nullcontext
from trl.trainer.grpo_trainer import nanstd
from trl.models import unwrap_model_for_generation
from trl.extras.profiling import profiling_context
from vllm.sampling_params import GuidedDecodingParams
from accelerate.utils import gather_object, broadcast_object_list
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from trl.data_utils import maybe_apply_chat_template, is_conversational
import re
import pandas as pd

# Fix VLLM compatibility issue - force V0 engine before importing VLLM
# os.environ["VLLM_USE_V1"] = "0"

sys.path.insert(0, os.environ['ROOT_PATH'])
print(sys.path)
import re
import time
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset, concatenate_datasets, Dataset, disable_caching
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer, PPOConfig, PPOTrainer, get_peft_config, ModelConfig
import datasets
from math_utils import process_docs, process_result_v1
# Task-specific imports
from blocksworld_reward_model import BlocksWorldModel
from utils import generate_icl, sc_output_extractor
from reasoners.benchmark import BWEvaluator
from reasoners.lm import HFModel
import numpy as np
# For countdown task
import random
from countdown_reward_model import CountdownRewardModel
import math
from functools import partial
# For Arithmetic Tasks
from gsm8k_reward_model import GSM8KRewardModel
import logging
from accelerate import Accelerator
from vllm import LLM, SamplingParams
# For Coding Tasks
from coding_reward_model import CodingRewardModel
# For Variance Regularized Scheduler
from methods.RL.schedulers.variance_regularized_scheduler import _variance_regularized_schedule, update_variance_regularized_performance_v2, reset_variance_regularized_state
import torch.distributed as dist
log = logging.getLogger(__name__)
OmegaConf.register_new_resolver("d2s", lambda digit, sub: str(digit).replace(".", "_"))
OmegaConf.register_new_resolver("mode2name", lambda mode, sub1, sub2: sub1 if mode == "train" else sub2)
OmegaConf.register_new_resolver("ckpt2short", lambda ckpt: f"_FT{ckpt.split('_')[-1]}" if ckpt else "")

disable_caching()
accelerator = Accelerator()
def log_on_main(text):
    if accelerator.is_main_process:
        log.info(text)

class TaskSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_tasks, total_iterations, data_schedule, batch_size, mini_repeat_count, repeat_count, scheduler_params, resample_size, seed=0, trainer=None):
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
        task_col = np.asarray(self.dataset['task'])
        self.indices_by_task = {
            t: self.rng.permutation(np.where(task_col == t)[0])
            for t in range(self.num_tasks)
        }
        self.schedule_funcs = {
            'balanced': self._balanced_schedule,
            'cosine': self._cosine_schedule,
            'gaussian': partial(self._gaussian_schedule, **scheduler_params),
            'classic': self._step_schedule,
            'vrex': partial(_variance_regularized_schedule, trainer=trainer, **scheduler_params)
        }
        log_on_main(f"Data Schedule: {data_schedule}")
        self.schedule_func = self.schedule_funcs[data_schedule]
        self.resample_size = resample_size
        self.task_ptrs = {t: 0 for t in range(self.num_tasks)}
        self.current_iteration = None
    
    # Classical Curriculum Learning
    @staticmethod    
    def _step_schedule(t, T, num_tasks):
        active_task = min(int(t * num_tasks / T), num_tasks - 1)
        return dict(enumerate(np.eye(num_tasks)[active_task].tolist()))

    def __iter__(self):
        
        indices_by_task = {t: idx.copy() for t, idx in self.indices_by_task.items()}
        
        # Don't reset variance regularized state here - it's done once in trainer init
        
        for i in range(self.total_iterations):
            self.current_iteration = i
            probs_dict = self.schedule_func(i, self.total_iterations, self.num_tasks)
            print(f"[VREx Sampler DEBUG] Iteration {i}: Schedule func returned: {probs_dict}")
            probs = np.array([probs_dict[j] for j in range(self.num_tasks)])
            # Sample a task for each slot in the batch using the probabilities.
            chosen_tasks = self.rng.choice(np.arange(self.num_tasks), size=self.batch_size, p=probs, replace=True)
            # if rank == 0:
            batch_indices = []

            for t in chosen_tasks:
                bucket = self.indices_by_task[int(t)]
                p = self.task_ptrs[int(t)] % len(bucket)  # wrap
                batch_indices.append(int(bucket[p]))
                self.task_ptrs[int(t)] += 1
            
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
    
    def _draw_group_indices(self, n_groups: int) -> list[int]:
        """Draw n_groups dataset indices using the curriculum at current_iteration."""
        probs_dict = self.schedule_func(self.current_iteration, self.total_iterations, self.num_tasks)
        probs = np.array([probs_dict[j] for j in range(self.num_tasks)], dtype=np.float64)
        probs /= probs.sum()
        chosen_tasks = self.rng.choice(np.arange(self.num_tasks), size=int(n_groups), p=probs, replace=True)
        print(f'Dapo Resample Chosen Tasks: {chosen_tasks} - iteration: {self.current_iteration}')
        picked = []
        for t in chosen_tasks:
            bucket = self.indices_by_task[int(t)]
            p = self.task_ptrs[int(t)] % len(bucket)   # rolling pointer so we don’t repeat same rows
            picked.append(int(bucket[p]))
            self.task_ptrs[int(t)] += 1
        return picked
    
    def resample(self, k = None) -> list[int]:
        """
        Redraw EXACTLY n_groups prompt indices for this same curriculum step.
        Needed for DAPO.
        """
        k = k if k is not None else self.resample_size
        base = self._draw_group_indices(k)  
        repeated = [idx for idx in base for _ in range(self.mini_repeat_count)]
        return repeated

class CurriculumGRPOTrainer(GRPOTrainer):
    def __init__(self, num_tasks=4, total_iterations=1200, data_schedule='balanced', scheduler_params: dict=None, *args, **kwargs):
        self.num_tasks = num_tasks
        self.total_iterations = total_iterations
        self.data_schedule = data_schedule
        self.scheduler_params=scheduler_params
        # self.data2reward_fn_name = {'countdown': '_countdown_reward_fn', 'blocksworld'}
        # Reset variance regularized state once at initialization
        if self.data_schedule == 'vrex':
            reset_variance_regularized_state()
        self.max_dapo_iter = 0
        self._last_batch_rewards = None
        self._task_sampler = None
        self._compute_loss_log = {}
        super().__init__(*args, **kwargs)

    def _check_if_adv_zero(self):
        # Check if advantage is zero to enable DAPO sampling
        if not self._last_batch_rewards: return False
        r = np.array(self._last_batch_rewards)
        rewards_per_prompts = r.reshape(-1, self.num_generations)
        print('Rewards per prompts:', rewards_per_prompts)
        eps = 1e-6
        zero_adv = np.std(rewards_per_prompts, axis=1, ddof=0) <= eps
        print(f'Was there Zero ADV? {zero_adv} - {self._task_sampler.current_iteration}')
        return bool(np.any(zero_adv))

    def _get_train_sampler(self, train_dataset=None):
        # The parent class passes the dataset as an argument, but we use self.train_dataset

        # generation_batch_size = self.accelerator.num_processes (num_device) * self.args.per_device_train_batch_size (including num_generation)
        # * self.steps_per_generation (steps_per_generation is mutual exclusive to generation_batch_size)
        sampler = TaskSampler(self.train_dataset,
                           num_tasks=self.num_tasks,
                           total_iterations=self.total_iterations,
                           data_schedule=self.data_schedule,
                           scheduler_params=self.scheduler_params,
                           batch_size=self.args.generation_batch_size * self.args.gradient_accumulation_steps // self.num_generations,
                           mini_repeat_count=self.num_generations,
                           repeat_count=self.num_iterations,
                           resample_size=self.args.generation_batch_size // (self.accelerator.num_processes * self.num_generations),# * self.args.steps_per_generation, #num_iterations=1 is a GRPO param.
                           trainer=self)
        self._task_sampler = sampler
        return sampler

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
            # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
            # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            prompts_text = [
                re.sub(rf"^({re.escape(self.processing_class.pad_token)})+", "", text) for text in prompts_text
            ]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "guided_decoding": guided_decoding,
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ):
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        #############################################################################
        # Moved From Line 393
        #############################################################################
        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)
        #############################################################################
        # Moved From Line 393
        #############################################################################

        #############################################################################
        # Added
        #############################################################################
        if mode == "train" and is_std_zero.any():
            current_dapo_iter = getattr(self, "current_dapo_iter", 0)
            max_dapo_iter = self.scheduler_params.max_dapo_iter
            if current_dapo_iter < max_dapo_iter:
                print(f"Dynamic Sampling (iter {current_dapo_iter+1}/{max_dapo_iter}): Found {is_std_zero.sum().item()}/{len(is_std_zero)} groups with zero std. Resampling batch...")
                self.current_dapo_iter = current_dapo_iter + 1
                resampled_indices = self._task_sampler.resample()
                resampled_inputs = [self.train_dataset[int(idx)] for idx in resampled_indices]
                # TODO: Resample (self.args.generation_batch_size // self.num_generations) New prompts.
                # TODO: Inputs will be a list of size 16. (2 new prompts, repeated 8 times).
                # TODO: Make resample func in sampler that directly gives these prompts
                result = self._generate_and_score_completions(resampled_inputs)
                self.current_dapo_iter = 0
                return result
            else:
                print(f"Dynamic Sampling: Max iterations ({max_dapo_iter}) reached.")
                self.current_dapo_iter = 0
        print(f"Dynamic Sampling: Found {is_std_zero.sum().item()}/{len(is_std_zero)} groups with zero std. Proceeding with batch...")
        #############################################################################
        # Added
        #############################################################################

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, logits_to_keep
                        )
            else:
                ref_per_token_logps = None

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Extract task IDs from the batch before processing
        if 'task' in inputs[0]:
            self._current_batch_task_ids = [inp['task'] for inp in inputs]
        
        # Call parent training step
        result = super().training_step(model, inputs, num_items_in_batch)
        
        # Update variance regularized scheduler if using it
        if self.data_schedule == 'vrex' and hasattr(self, '_last_batch_rewards') and hasattr(self, '_current_batch_task_ids'):
            advantages = list(self._textual_logs['advantages'])
            task_ids = self._current_batch_task_ids
            rewards = self._last_batch_rewards    # can use self._textual_logs['rewards']['_countdown_reward_fn'] if we can somehow know which dataset we are using here.
            update_variance_regularized_performance_v2(task_ids, rewards, advantages=advantages, trainer=self)
            
            # Log VREx metrics at the correct time - AFTER training step, BEFORE log_stats buffer clear
            # Only log when it aligns with logging_steps
            if hasattr(self, '_vrex_metrics_to_log'):# and self.state.global_step % self.args.logging_steps == 0:
                try:
                    # Use the trainer's log method which stages metrics for next log_stats call
                    # self.log(self._vrex_metrics_to_log)
                    for key, value in self._vrex_metrics_to_log.items():
                        self._metrics['train'][key].append(value)
                    # print(f"[VREx DEBUG] Successfully logged {len(self._vrex_metrics_to_log)} metrics after training step")
                    # Clear the stored metrics
                    delattr(self, '_vrex_metrics_to_log')
                except Exception as e:
                    print(f"[VREx ERROR] Failed to log stored metrics: {e}")
            
            # Clean up
            delattr(self, '_current_batch_task_ids')
        
        return result

# class Cosine

class BaseTrainer:
    """Base class for training and inference with Hydra configuration"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # Setting up paths
        self.output_dir = Path(HydraConfig.get().run.dir)  # Hydra changes working directory

        # Save the config for reproducibility
        with open(self.output_dir / "config_dump.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

        root_path = Path(os.environ['ROOT_PATH'])
        os.chdir(root_path)
        print(f"Working directory: {root_path}")
        print(f"Output directory: {self.output_dir}")

        # Setup HuggingFace authentication
        # Setup HuggingFace authentication
        hf_token = self.cfg.experiment.hf_token
        # Check if already logged in using huggingface_hub API
        from huggingface_hub import HfApi
        try:
            # Try to get user info which will fail if not logged in
            api = HfApi()
            user_info = api.whoami()
            print(f"Already logged in to Hugging Face as {user_info['name']}")
        except Exception as e:
            print("Logging in to Hugging Face")
            login(token=hf_token, add_to_git_credential=True)

        self.last_log_time = None

    def train(self):
        """Train a model"""
        raise NotImplementedError("Train method must be implemented by subclasses")

    def inference(self):
        """Run inference"""
        raise NotImplementedError("Inference method must be implemented by subclasses")

    def _get_model_config(self):
        """Create model configuration"""
        lora_config = self.cfg.lora

        model_config = ModelConfig(
            model_name_or_path=self.cfg.model.name,
            torch_dtype=self.cfg.model.torch_dtype,
            attn_implementation=self.cfg.model.attn_implementation,
            lora_task_type=lora_config.task_type,
            lora_r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            lora_target_modules=list(lora_config.target_modules),
        )

        return model_config

    def _get_checkpoint_path(self, checkpoint, model_name=None):
        """Generate the path to the checkpoint based on configuration"""
        # Get the base output directory for models from config
        base_output_dir = self.cfg.output.root_path
        # Use the model name from output config
        model_name = self.cfg.output.run_name if model_name is None else model_name

        # If the checkpoint is a number, use the checkpoint-{num} format
        if isinstance(checkpoint, int):
            checkpoint_path = os.path.join(base_output_dir, "outputs", model_name,
                                           f"checkpoint-{checkpoint}")
        else:
            # Otherwise use the provided checkpoint path directly
            checkpoint_path = checkpoint

        # Ensure the checkpoint exists
        if os.path.exists(checkpoint_path):
            return checkpoint_path
        else:
            import warnings

            warnings.warn(f"Checkpoint not found at {checkpoint_path}.")
            checkpoint_path = f'{self.cfg.model.family}/{model_name}'
            warnings.warn(f"Will attempt to use the model in huggingface: {checkpoint_path}.")


        return checkpoint_path

    def _get_common_training_args(self):
        """Get common training arguments for both GRPO and PPO"""
        training_cfg = self.cfg.algorithm.training
        output_dir = self.output_dir

        common_args = {
            "learning_rate": training_cfg.learning_rate,
            "lr_scheduler_type": training_cfg.lr_scheduler_type,
            "logging_steps": training_cfg.logging_steps,
            "max_steps": training_cfg.max_steps * len(self.cfg.task.data_files) if training_cfg.curriculum else training_cfg.max_steps,
            "per_device_train_batch_size": training_cfg.per_device_train_batch_size,
            "gradient_accumulation_steps": training_cfg.gradient_accumulation_steps,
            "gradient_checkpointing": training_cfg.gradient_checkpointing,
            "bf16": training_cfg.bf16,
            "report_to": list(training_cfg.report_to),
            "push_to_hub": training_cfg.push_to_hub,
            "save_strategy": training_cfg.save_strategy,
            "save_steps": training_cfg.save_steps,
            "tf32": training_cfg.tf32,
            "output_dir": str(output_dir),
            "run_name": self.cfg.output.run_name,
            "hub_model_id": self.cfg.output.run_name,
            "seed": self.cfg.experiment.dataset_seed,
            "logging_dir": str(output_dir),
            "eval_strategy": "no",
            "accelerator_config": {'split_batches': True}
        }

        return common_args, output_dir

    def _setup_grpo_training(self):
        """Setup training configuration for GRPO"""
        training_cfg = self.cfg.algorithm.training
        common_args, _ = self._get_common_training_args()

        # Add GRPO specific parameters
        grpo_args = {
            # GRPO specific parameters
            "max_prompt_length": self.cfg.task.training.max_prompt_length,
            "max_completion_length": self.cfg.task.training.max_completion_length,
            "steps_per_generation": training_cfg.steps_per_generation,
            "generation_batch_size": training_cfg.generation_batch_size,
            "num_generations": training_cfg.num_generations,
            "beta": training_cfg.beta,
            # Vllm
            "use_vllm": training_cfg.use_vllm,
            "vllm_mode": training_cfg.vllm_mode,
            "vllm_tensor_parallel_size": 1,
            # "vllm_server_port": 12110,# Add missing vllm_mode parameter
            "vllm_gpu_memory_utilization": training_cfg.vllm_gpu_memory_utilization,
        }
        if training_cfg.vllm_mode == 'server':
            grpo_args["vllm_server_port"] = training_cfg.vllm_server_port
        # Combine common and GRPO specific args
        training_args = GRPOConfig(**common_args, **grpo_args)

        return training_args

    def _setup_ppo_training(self):
        """Setup training configuration for PPO"""
        training_cfg = self.cfg.algorithm.training
        common_args, _ = self._get_common_training_args()

        # Add PPO specific parameters
        ppo_args = {
            # PPO specific parameters
            "num_ppo_epochs": training_cfg.num_ppo_epochs,
            "kl_coef": training_cfg.kl_coef,
            "cliprange": training_cfg.cliprange,
            "vf_coef": training_cfg.vf_coef,
            "cliprange_value": training_cfg.cliprange_value,
            "gamma": training_cfg.gamma,
            "lam": training_cfg.lam,
            "whiten_rewards": training_cfg.whiten_rewards,
        }

        # Combine common and PPO specific args
        training_args = PPOConfig(**common_args, **ppo_args)

        return training_args


class BlocksWorldTrainer(BaseTrainer):
    """Class for training and inference on blocksworld models"""

    def _prepare_dataset(self):
        """Prepare dataset for training"""
        # If a dataset size limit is specified, sample equally from each file
        all_samples = []
        data_files = self.cfg.task.data_files
        data_schedule = self.cfg.algorithm.training.curriculum_schedule
        for task_idx, file in enumerate(data_files):
            file_dataset = load_dataset('json', data_files=file)['train']
            file_dataset = file_dataset.shuffle(seed=self.cfg.experiment.dataset_seed)
            
            # if self.cfg.experiment.dataset_size > 0 and data_schedule == 'fixed':
            #     num_files = len(data_files)
            #     samples_per_file = self.cfg.experiment.dataset_size // num_files
            #     num_samples = min(len(file_dataset), samples_per_file)
            #     file_dataset = file_dataset.select(range(num_samples))
            
            # Annotate with difficulty
            task_annotations = [task_idx] * len(file_dataset)
            file_dataset = file_dataset.add_column("task", task_annotations)
            
            all_samples.extend(file_dataset)
        dataset = Dataset.from_list(all_samples)

        dataset = dataset.shuffle(seed=self.cfg.experiment.dataset_seed)
        print(f"Dataset prepared with {len(dataset)} samples")
        return dataset

    def _generate_prompt(self, tokenizer, init, goal, plan="", example_index=0, icl_examples_set=None):
        """Generate prompt for the blocksworld model"""
        if icl_examples_set is None:
            icl_example = ""
        else:
            icl_example = generate_icl(icl_examples_set, provide_think_icl=True, num_icl=1, idx=example_index)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
            },
            {
                "role": "user",
                "content": f"I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nHere is the format of the actions: \n\npick up the [block_name] block # for example: pick up the blue block\nunstack the [block_name] block from on top of the [another_block_name] block # for example: unstack the orange block from on top of the black block\nput down the [block_name] block # for example put down the red block\nstack the [block_name] block on top of the [another_block_name] block # for example: stack the yellow block on top of the red block \n\n{icl_example}\n\n[Problem]\nHere is the initial state of the blocks: {init}\n\nHere is the goal state of the blocks: {goal}. Show your work in <think> </think> tags. After that, provide the final answer in <answer> </answer> tags, for example <answer>\nunstack the cyan block from on top of the emerald block\nput down the cyan block</answer>\n"
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>"
            }
        ]

        return {
            "prompt": tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True),
            "plan": plan,
            "init": init,
            "goal": goal
        }

    def _validate_bw_response_format(self, response: str):
        """Validate the blocksworld response format"""
        # Remove leading/trailing whitespace
        response = response.strip()

        # Rule 1: Must start with <think> and end with </plan>
        if not response.startswith("<think>") or not response.endswith("</answer>"):
            print('Response does not start with <think> or end with </answer>')
            return False

        # Rule 2: Must contain exactly one of each tag.
        if response.count("<think>") != 1 or response.count("</think>") != 1:
            print('Response does not contain exactly one of each think tag')
            return False
        if response.count("<answer>") != 1 or response.count("</answer>") != 1:
            print('Response does not contain exactly one of each answer tag')
            return False

        # Find indices for each tag.
        think_open = response.find("<think>")
        think_close = response.find("</think>")
        plan_open = response.find("<answer>")
        plan_close = response.find("</answer>")

        # Rule 4: The order should be: <think> ... </think> then <answer> ... </answer>
        if think_open != 0:  # Should start with <think>
            print('Response does not start with <think>')
            return False
        if think_close == -1 or plan_open == -1 or plan_close == -1:
            print('Response does not contain <answer> and </answer>, or </think>')
            return False
        if think_close > plan_open:
            print('Response has closing think tag after opening answer tag')
            return False  # The closing think tag must come before the opening plan tag

        # Rule 3: Check non-empty content between tags.
        think_content = response[len("<think>"):think_close].strip()
        plan_content = response[plan_open + len("<answer>"):plan_close].strip()

        if not think_content or not plan_content:
            return False

        return True

    def _blocksworld_reward_fn(self, completions, plan, init, goal, **kwargs):
        """Reward function for blocksworld task"""
        rewards = []
        for completion, plan_i, init_i, goal_i in zip(completions, plan, init, goal):
            reward_format = 0.0
            try:
                print('#########################')
                completion = "<think>" + completion
                print(completion)

                if not self._validate_bw_response_format(completion):
                    print('Response Format Error')
                    rewards.append(0.0)  # Penalty to avoid format errors
                    continue
                else:
                    reward_format = 1.0

                # Extract the plan
                matches = re.findall(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL | re.IGNORECASE)
                if matches is None or len(matches) != 1:
                    print("No plan found")
                    rewards.append(0.0)
                    continue

                # Process plan
                non_empty = [match.strip() for match in matches if
                             match.strip()]  # Ideally, we should have only one match
                extracted_plan = non_empty[0]

                # Calculate reward
                instance_example = BlocksWorldModel(init_i, goal_i, extracted_plan)
                reward = instance_example.simulate_plan_with_reward(true_plan=plan_i) + reward_format
                rewards.append(reward)
                print('-----')
                print(reward)
                print(init_i)
                print(goal_i)
                print('-----')
                print('#########################')
            except Exception as e:
                print(e)
                rewards.append(0.0)

        # Update variance regularized scheduler if we're in training mode
        # and task IDs are available
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'data_schedule'):
            if self.trainer.data_schedule == 'vrex':
                # Store rewards in trainer for later use
                self.trainer._last_batch_rewards = rewards

        return rewards

    def train(self):
        """Train a model using the specified algorithm with configurations from Hydra"""
        # Extract config values
        model_name = self.cfg.model.name
        use_icl_examples = self.cfg.task.use_icl_examples
        output_model_name = self.cfg.output.run_name
        algorithm = self.cfg.algorithm.name

        # Prepare ICL examples if needed
        icl_examples = None
        if use_icl_examples:
            with open(self.cfg.task.icl_examples_file) as f:
                icl_examples = json.load(f)

        # Load tokenizer and model
        model_config = self._get_model_config()
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path,
            trust_remote_code=model_config.trust_remote_code
        )

        # Ensure we have a pad_token
        if tokenizer.pad_token is None:
            # Option A: alias EOS → PAD
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            torch_dtype=model_config.torch_dtype,
            trust_remote_code=model_config.trust_remote_code,
            attn_implementation=model_config.attn_implementation
        )
        peft_config = get_peft_config(model_config)

        # Prepare dataset
        dataset = self._prepare_dataset()
        dataset = dataset.map(
            lambda example, idx: self._generate_prompt(
                tokenizer,
                example["init"],
                example["goal"],
                example["plan"],
                idx,
                icl_examples
            ),
            with_indices=True
        )

        # Split dataset
        train_test_split = dataset.train_test_split(test_size=self.cfg.experiment.test_size)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]

        # Setup training arguments based on algorithm
        if 'grpo' in algorithm:
            training_args = self._setup_grpo_training()
            trainer = CurriculumGRPOTrainer(
                model=model,
                reward_funcs=self._blocksworld_reward_fn,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                processing_class=tokenizer,
                peft_config=peft_config,
                num_tasks=len(self.cfg.task.data_files),
                total_iterations=training_args.max_steps,
                data_schedule=self.cfg.algorithm.training.curriculum_schedule,
                scheduler_params=self.cfg.algorithm.training.scheduler_params,
            )
            # Store trainer reference for reward function access
            self.trainer = trainer

        elif algorithm == "ppo":
            training_args = self._setup_ppo_training()
            trainer = PPOTrainer(
                model=model_config.model_name_or_path,
                ref_model=model_config.model_name_or_path,  # Same model as reference
                tokenizer=tokenizer,
                args=training_args,
                reward_fn=self._blocksworld_reward_fn,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                peft_config=get_peft_config(model_config),
            )

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Train model
        trainer.train()
        trainer.save_model(training_args.output_dir)

        if self.cfg.algorithm.training.push_to_hub:
            trainer.push_to_hub(dataset_name='blocksworld-dataset')

    def _train_ppo(self, trainer, dataset, tokenizer):
        """Custom training loop for PPO"""
        print("Starting PPO training loop for BlocksWorld task")

        # Use smaller subset during PPO training due to computational constraints
        if len(dataset) > 100:
            train_dataset = dataset.select(range(100))
        else:
            train_dataset = dataset

        for epoch in range(self.cfg.algorithm.training.max_steps):
            print(f"PPO Epoch {epoch}/{self.cfg.algorithm.training.max_steps}")

            # Sample batch of prompts
            # Sample batch of prompts - use the per_device_train_batch_size as batch size
            batch_indices = random.sample(range(len(train_dataset)),
                                          min(self.cfg.algorithm.training.per_device_train_batch_size,
                                              len(train_dataset)))
            batch = [train_dataset[i] for i in batch_indices]

            # Prepare inputs
            query_tensors = []
            for item in batch:
                input_ids = tokenizer(item["prompt"], return_tensors="pt").input_ids
                if hasattr(trainer, "accelerator"):
                    input_ids = input_ids.to(trainer.accelerator.device)
                query_tensors.append(input_ids)

            # Generate model responses
            response_tensors = []
            for query in query_tensors:
                response = trainer.generate(
                    query,
                    max_new_tokens=self.cfg.task.training.max_completion_length,
                    do_sample=True,
                    temperature=0.7
                )
                response_tensors.append(response)

            # Compute rewards
            rewards = []
            for i, (response, item) in enumerate(zip(response_tensors, batch)):
                # Decode the response
                response_text = tokenizer.decode(response[0], skip_special_tokens=True)

                # Extract the completion part (after "<think>")
                if "<think>" in response_text:
                    completion = response_text.split("<think>")[1]
                else:
                    completion = response_text

                # Compute reward using the blocksworld reward function
                reward = self._blocksworld_reward_fn(
                    [completion],
                    [item["plan"]],
                    [item["init"]],
                    [item["goal"]]
                )[0]

                rewards.append(reward)
                print(f"Sample {i}, Reward: {reward}")

            # Convert rewards to tensors
            reward_tensors = [torch.tensor(reward) for reward in rewards]

            # Perform PPO update
            stats = trainer.step(query_tensors, response_tensors, reward_tensors)

            # Log training progress
            if epoch % self.cfg.algorithm.training.logging_steps == 0:
                print(f"Epoch {epoch}: {stats}")

                # Save checkpoint
                if epoch % self.cfg.algorithm.training.save_steps == 0:
                    trainer.save_pretrained(f"{trainer.args.output_dir}/checkpoint-{epoch}")

    def inference(self):
        """Run inference using the trained model"""
        # Extract config values
        model_checkpoint = self.cfg.task.inference.checkpoint
        steps = self.cfg.task.inference.steps
        temperature = self.cfg.task.inference.temperature
        sc_num = self.cfg.task.inference.sc_num
        pass_at_k = self.cfg.task.inference.pass_at_k
        use_icl = self.cfg.task.inference.use_icl
        prompt_path = self.cfg.task.inference.prompt_path
        resume = self.cfg.task.inference.resume
        mode = 'pass' if pass_at_k == 1 else 'majority'
        # Generate checkpoint path
        model_dir = self._get_checkpoint_path(model_checkpoint)
        max_batch_size = self.cfg.algorithm.inference.max_batch_size
        # Setup data path
        data_path = self.cfg.task.inference.data_path.format(steps=steps)

        # Setup log directory
        model_name = model_dir.split('/')[-1]
        log_dir = f'logs/Blocksworld/RL/step_{steps}/{datetime.now().strftime("%m%d%Y-%H%M%S")}_{model_name}_t_{temperature}_sc_{sc_num}'

        # Load prompt
        with open(prompt_path) as f:
            prompt = json.load(f)

        # Prepare ICL examples if needed
        icl = ""
        if use_icl:
            with open(self.cfg.task.icl_examples_file) as f:
                icl_examples = json.load(f)
            icl = generate_icl(icl_examples, provide_think_icl=True, num_icl=self.cfg.task.inference.icl_num)
        print(f"ICL examples: {icl}")

        # Load model
        base_model = HFModel(
            model_pth=model_dir,
            tokenizer_pth=model_dir,
            max_new_tokens=self.cfg.task.inference.max_new_tokens,
            max_batch_size=max_batch_size
        )

        # Create reasoner
        reasoner = RLReasoner(
            base_model,
            temperature=temperature,
            sc_num=sc_num,
            icl_example=icl,
            pass_at_k=pass_at_k,
        )

        # Setup evaluator
        evaluator = BWEvaluator(
            config_file=self.cfg.task.inference.config_file,
            domain_file=self.cfg.task.inference.domain_file,
            data_path=data_path,
            init_prompt=prompt,
            disable_log=False,
            output_extractor=lambda x: sc_output_extractor(x, mode=mode),
            mode=mode,
            sample_prompt_type="rap"  # rap prompt includes cot
        )

        # Run evaluation
        accuracy = evaluator.batched_evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir, batch_size=max_batch_size)

        print(f'Accuracy: {accuracy}')

        # Save results to output directory
        results = {
            "accuracy": accuracy,
            "model_checkpoint": model_checkpoint,
            "steps": steps,
            "temperature": temperature,
            "sc_num": sc_num,
            "use_icl": use_icl,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(self.output_dir / "inference_results.json", "w") as f:
            json.dump(results, f, indent=2)

        return accuracy


class CountdownTrainer(BaseTrainer):
    """Class for training and inference on countdown models"""

    def _prepare_dataset(self, data_files):
        """Prepare dataset for training"""
        if self.cfg.task.force_redownload:
            all_data = [load_dataset(data_path, download_mode='FORCE_REDOWNLOAD') for data_path in data_files]
        else:
            all_data = [load_dataset(data_path) for data_path in data_files]
        train_data = [data['train'] for data in all_data]
        test_data = [data['test'] for data in all_data]
        # Annotate with difficulty
        add_task_difficulty = lambda task_idx, dataset: dataset.add_column("task", [task_idx] * len(dataset))
        train_data = [add_task_difficulty(i, data) for i, data in enumerate(train_data)]
        test_data = [add_task_difficulty(i, data) for i, data in enumerate(test_data)]

        train_dataset = concatenate_datasets(train_data)
        test_dataset = concatenate_datasets(test_data)
        train_dataset = train_dataset.shuffle(seed=self.cfg.experiment.dataset_seed)
        test_dataset = test_dataset.shuffle(seed=self.cfg.experiment.dataset_seed)

        # Limit dataset size if specified
        if self.cfg.task.train_size > 0:
            train_dataset = train_dataset.select(range(self.cfg.task.train_size))
            test_dataset = test_dataset.select(range(self.cfg.task.test_size))

        print(f"Dataset prepared with {len(train_dataset)} training samples and {len(test_dataset)} test samples")
        return train_dataset, test_dataset

    def _construct_reasoning_trace(self, reasoning_steps):
        """Construct reasoning trace from reasoning steps"""
        reasoning_trace = []
        n_r = len(reasoning_steps) - 1
        for i, step in enumerate(reasoning_steps):
            if 0 < i < n_r:
                reasoning_trace.append(f"Step {i}: {step}")
        reasoning_trace.append(f"Final Result: {reasoning_steps[-1]}")
        return reasoning_trace

    def _generate_prompt(self, tokenizer, example):
        """Generate prompt for the countdown model"""
        # Extract target and numbers from the example
        data = example.get("reward_model", {}).get("ground_truth", {})
        target = data.get("target")
        numbers = data.get("numbers")
        # expression = data.get("expression") # e.g., (((76 - 80) - 28) + 43), (((65 * 12) + 60) / 28)
        reasoning_steps = example.get("reasoning_steps")
        reasoning_trace = self._construct_reasoning_trace(reasoning_steps)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
            },
            {
                "role": "user",
                "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>"
            }
        ]

        return {
            "prompt": tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True),
            "target": target,
            "numbers": numbers,
            "reasoning_trace": reasoning_trace,
        }

    def _validate_countdown_response_format(self, response: str):
        """Validate the countdown response format"""
        # Remove leading/trailing whitespace
        response = response.strip()

        # Must contain <think> and </think> tags
        if "<think>" not in response or "</think>" not in response:
            print('Response does not contain think tags')
            return False

        # Must contain <answer> and </answer> tags
        if "<answer>" not in response or "</answer>" not in response:
            print('Response does not contain answer tags')
            return False

        # Check that tags are in correct order
        think_open = response.find("<think>")
        think_close = response.find("</think>")
        answer_open = response.find("<answer>")
        answer_close = response.find("</answer>")

        if think_close < think_open or answer_close < answer_open:
            return False

        if answer_open < think_close:
            return False

        return True

    def _countdown_reward_fn(self, completions, target, numbers, **kwargs):
        """Reward function for countdown task"""
        rewards = []

        # Check if task IDs are provided in kwargs
        task_ids = kwargs.get('task_ids', None)

        for completion, target_i, numbers_i in zip(completions, target, numbers):
            try:
                print('#########################')
                completion = "<think>" + completion
                print(completion)

                if not self._validate_countdown_response_format(completion):
                    print('Response Format Error')
                    rewards.append(0.0)  # Penalty to avoid format errors
                    continue

                # Use the CountdownRewardModel class
                reward_model = CountdownRewardModel(target_i, numbers_i)
                reward = reward_model.compute_score(completion)
                rewards.append(reward)
                print('-----')
                print(reward)
                print(target_i)
                print(numbers_i)
                print('-----')
                print('#########################')
            except Exception as e:
                print(e)
                rewards.append(0.0)

        # Update variance regularized scheduler if we're in training mode
        # and task IDs are available
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'data_schedule'):
            if self.trainer.data_schedule == 'vrex':
                # Store rewards in trainer for later use
                self.trainer._last_batch_rewards = rewards

        return rewards

    def train(self):
        """Train a model using the specified algorithm with configurations from Hydra"""
        # Extract config values
        model_name = self.cfg.model.name
        output_model_name = self.cfg.output.run_name
        algorithm = self.cfg.algorithm.name
        
        # Check if we're loading from checkpoint
        checkpoint_path = getattr(self.cfg.algorithm.training, 'resume_from_checkpoint', None)

        # Load tokenizer and model
        model_config = self._get_model_config()
        
        # Determine model path - either checkpoint or base model
        if checkpoint_path:
            model_path = self._get_checkpoint_path(int(checkpoint_path.split('_')[-1]), checkpoint_path)
            print(f"Loading model from checkpoint: {model_path}")
        else:
            model_path = model_config.model_name_or_path
            print(f"Loading base model: {model_path}")
            
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=model_config.trust_remote_code
        )
        # Ensure we have a pad_token
        if tokenizer.pad_token is None:
            # Option A: alias EOS → PAD
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=model_config.torch_dtype,
            trust_remote_code=model_config.trust_remote_code,
            attn_implementation=model_config.attn_implementation
        )
        peft_config = get_peft_config(model_config)

        # Prepare dataset
        train_dataset, test_dataset = self._prepare_dataset(self.cfg.task.data_files)
        train_dataset = train_dataset.map(lambda example: self._generate_prompt(tokenizer, example))
        test_dataset = test_dataset.map(lambda example: self._generate_prompt(tokenizer, example))

        # Split dataset
        # train_test_split = dataset.train_test_split(test_size=self.cfg.task.test_size)
        # train_dataset = train_test_split["train"]
        # test_dataset = train_test_split["test"]

        # Setup training arguments based on algorithm
        if "grpo" in algorithm:
            training_args = self._setup_grpo_training()
            trainer = CurriculumGRPOTrainer(
                model=model,
                reward_funcs=self._countdown_reward_fn,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                processing_class=tokenizer,
                peft_config=peft_config,
                num_tasks=len(self.cfg.task.data_files),
                total_iterations=training_args.max_steps,
                data_schedule=self.cfg.algorithm.training.curriculum_schedule,
                scheduler_params=self.cfg.algorithm.training.scheduler_params,
            )
            # Store trainer reference for reward function access
            self.trainer = trainer

        elif algorithm == "ppo":
            training_args = self._setup_ppo_training()
            trainer = PPOTrainer(
                model=model_config.model_name_or_path,
                ref_model=model_config.model_name_or_path,  # Same model as reference
                tokenizer=tokenizer,
                args=training_args,
                reward_fn=self._countdown_reward_fn,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                peft_config=get_peft_config(model_config),
            )

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Train model
        trainer.train()
        trainer.save_model(training_args.output_dir)

        if self.cfg.algorithm.training.push_to_hub:
            trainer.push_to_hub(dataset_name='countdown-dataset')

    def inference(self):
        """Run inference using the trained model"""
        # Extract config values
        model_checkpoint = self.cfg.task.inference.checkpoint
        sc_num = self.cfg.task.inference.sc_num
        pass_at_k = self.cfg.task.inference.pass_at_k
        assert not (sc_num > 1 and pass_at_k > 1), "sc_num > 1 and pass_at_k > 1 is not supported"
        num_generations = pass_at_k if pass_at_k > 1 else sc_num
        batch_size = self.cfg.task.inference.batch_size

        # Generate checkpoint path
        model_dir = self._get_checkpoint_path(model_checkpoint, self.cfg.model.trim)

        # Load test dataset
        _, test_dataset = self._prepare_dataset([self.cfg.task.test_file])

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=self.cfg.model.trust_remote_code
        )

        # Custom model for inference
        # model = HFModel(
        #     model_pth=model_dir,
        #     tokenizer_pth=model_dir,
        #     max_new_tokens=self.cfg.task.inference.max_new_tokens,
        #     max_batch_size=batch_size
        # )
        model = LLM(
            model=model_dir,
            # tokenizer=tokenizer,
            trust_remote_code=self.cfg.model.trust_remote_code,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=self.cfg.model.torch_dtype,
            gpu_memory_utilization=self.cfg.algorithm.training.vllm_gpu_memory_utilization,
            max_model_len=2048,
            seed=self.cfg.experiment.dataset_seed,
            task='generate'
        )

        tokenizer = model.get_tokenizer()
        if tokenizer.pad_token is None:
            # Option A: alias EOS → PAD
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        sampling_params = SamplingParams(
            n=num_generations,
            temperature=self.cfg.task.inference.temperature,
            max_tokens=self.cfg.task.inference.max_new_tokens,
            min_tokens=1,
            seed=self.cfg.experiment.dataset_seed,
            skip_special_tokens=False,
            top_p=0.9,
            top_k=50
        )

        # Run inference on test dataset
        correct = 0
        rewards = 0
        total = 0
        results = []

        if pass_at_k > 1:
            split_train_eval_dataset = test_dataset.train_test_split(train_size=100, seed=123, shuffle=False)
            test_dataset = split_train_eval_dataset["train"]

        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Testing batches"):
            prompt_data = [self._generate_prompt(tokenizer, test_dataset[k]) for k in range(i, min(i + batch_size, len(test_dataset)))]
            # Generate prompt
            numbers_list = [item["numbers"] for item in prompt_data]
            target_list = [item["target"] for item in prompt_data]
            prompt_list = [item["prompt"] for item in prompt_data]

            # Generate responses
            outputs = []
            output = model.generate(prompt_list, sampling_params)
            for i in range(len(prompt_list)):
                outputs.append([out.text for out in output[i].outputs])
            print(outputs)
            # print(outputs) # Qwen2.5-1.5B-Instruct_countdown2345_grpo_gaussian_0.25_0.75_True_1200
            # for _ in range(num_generations):
            #     outputs.append(model.generate(prompt_list, do_sample=True, temperature=self.cfg.task.inference.temperature, verbose=False, skip_special_tokens=False).text)
            # if num_generations > 1:
            #     # outputs = list(zip(*outputs)) # For old generation code
            #     pass
            # else:
            #     outputs = outputs[0]

            if pass_at_k > 1:
                for k_outputs, numbers, target, prompt in zip(outputs, numbers_list, target_list, prompt_list):

                    # Use the CountdownRewardModel for evaluation
                    reward_model = CountdownRewardModel(target, numbers)
                    pass_once = False
                    reward_per_sample = 0
                    result_per_sample = []
                    for output in k_outputs:
                        # Calculate score
                        score = reward_model.compute_score(output)

                        # Extract solution
                        solution = reward_model.extract_equation(output)

                        # Record results
                        # results.append({
                        #     "prompt": prompt,
                        #     "output": output,
                        #     "solution": solution,
                        #     "target": target,
                        #     "numbers": numbers,
                        #     "score": score
                        # })

                        if score > 0.5:  # Assuming score > 0.5 means correct answer
                            pass_once = True
                            result_per_sample.append(1)
                        else:
                            result_per_sample.append(0)
                        reward_per_sample += score
                    results.append(result_per_sample)
                    rewards += reward_per_sample / len(k_outputs)
                    correct += int(pass_once)
                    total += 1
            else:
                for k_outputs, numbers, target, prompt in zip(outputs, numbers_list, target_list, prompt_list):

                    # k = 1
                    output = k_outputs[0]

                    # Use the CountdownRewardModel for evaluation
                    reward_model = CountdownRewardModel(target, numbers)

                    # Calculate score
                    score = reward_model.compute_score(output)

                    # Extract solution
                    solution = reward_model.extract_equation(output)

                    # Record results
                    results.append({
                        "prompt": prompt,
                        "output": output,
                        "solution": solution,
                        "target": target,
                        "numbers": numbers,
                        "score": score
                    })

                    rewards += score
                    if score > 0.5:  # Assuming score > 0.5 means correct answer
                        correct += 1
                    total += 1

        if pass_at_k > 1:
            pd.DataFrame(results).to_csv(os.path.join(self.output_dir, f"pass_at_k_results_{self.cfg.task.test_file.split('/')[-1]}.csv"), index=False)
            # df_result = pd.DataFrame(results).transpose()
            # all_sample_pass_at_k_df = df_result.cummax(axis=0)
            # pass_at_k_df = df_result.mean(axis=1)
            # df_result.plot
        else:
            # Calculate accuracy
            accuracy = correct / total if total > 0 else 0
            rewards /= total if total > 0 else 0
            print(f'Accuracy: {accuracy}, Rewards: {rewards}')

            # Save results to output directory
            evaluation_results = {
                "accuracy": accuracy,
                "rewards": rewards,
                "model_checkpoint": model_checkpoint,
                "sc_num": sc_num,
                "detailed_results": results,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # with open(os.path.join(model_dir, "inference_results.json"), "w") as f:
            #     json.dump(evaluation_results, f, indent=2)

            return accuracy



class ArithmeticTrainer(BaseTrainer):
    """Class for training and inference on Arithmetic models"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.reward_functions = {
            'gsm8k': self.gsm8k_reward_fn,
            'aqua': self.aqua_reward_fn,
            'math': self.math_reward_fn
        }

    def _prepare_dataset(self):
        """Prepare dataset for training"""

        if self.cfg.mode == "train":
            data_files = self.cfg.task.training.data_files
        elif self.cfg.mode == "inference":
            data_files = self.cfg.task.inference.data_files

        datasetdict = {}
        for split in ['train', 'test']:
            dataset = []
            for task_idx, data_dir in enumerate(data_files):
                data = load_dataset('json', data_dir=data_dir, split=split)
                data = data.add_column("task", [task_idx] * len(data))
                dataset.append(data)
            dataset = concatenate_datasets(dataset)
            dataset = dataset.shuffle(seed=self.cfg.experiment.dataset_seed)            
            datasetdict[split] = dataset

        return datasets.DatasetDict(datasetdict)

    def _generate_prompt(self, tokenizer, example, use_icl = False):
        """Generate prompt for the arithmetic model"""

        if 'gsm8k' in self.cfg.task.name:
            question = example["question"]
            sft = example["solution"]
            answer = example["answer"]
            instruction = f"Solve the following math problem\n{question}\n\n Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 500 </answer>."

        elif 'aqua' in self.cfg.task.name:
            question = example["question"]
            sft = example["solution"]
            answer = example["answer"]
            options = "  ".join(example["options"])
            instruction = f"Solve the following math problem and choose an answer from the given options\n{question}\n{options}\n\n Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> C </answer>."

        elif 'math' in self.cfg.task.name:
            question = example["question"]
            sft = example["solution"]
            answer = example["answer"]
            instruction = f"Solve the following math problem\n{question}\n\nShow your work in <think> </think> tags. And return the final answer in \\boxed{{}}, wrapped in <answer> </answer> tags, for example <answer>\\boxed{500}</answer>."

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
            },
            {
                "role": "user",
                "content": instruction
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>"
            }
        ]

        return {
            "prompt": tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True),
            "sft" : sft,
            "answer": answer,
            "task" : example["task"]
        }

    @staticmethod
    def is_formatted(response: str):
        """Validate the response format"""
        response = response.strip()

        # Rule 1: Must start with <think> and end with </answer>
        if not response.startswith("<think>") or not response.endswith("</answer>"):
            return False, "Response does not start with <think> or end with </answer>"

        # Rule 2: Must contain exactly one of each tag.
        if response.count("<think>") != 1 or response.count("</think>") != 1:
            return False, 'Response does not contain exactly one of each think tag'
        if response.count("<answer>") != 1 or response.count("</answer>") != 1:
            return False, 'Response does not contain exactly one of each answer tag'

        # Find indices for each tag.
        think_open = response.find("<think>")
        think_close = response.find("</think>")
        plan_open = response.find("<answer>")
        plan_close = response.find("</answer>")

        # Rule 3: The order should be: <think> ... </think> then <answer> ... </answer>
        if think_open != 0:  # Should start with <think>
            return False, 'Response does not start with <think>'
        if think_close == -1 or plan_open == -1 or plan_close == -1:
            return False, 'Response does not contain <answer> and </answer>, or </think>'
        if think_close > plan_open:
            return False, 'Response has closing think tag after opening answer tag'

        # Rule 4: Check non-empty content between tags.
        think_content = response[len("<think>"):think_close].strip()
        plan_content = response[plan_open + len("<answer>"):plan_close].strip()
        if not think_content or not plan_content:
            return False, 'Empty content between tags'

        # Rule 5: Check <answer> immedietly follows </think>
        if not (response[think_close+len("</think>"):plan_open].strip() == ''):
            return False, 'There is content between </think> and <answer>'

        return True, 'Correctly Formatted'

    def math_reward_fn(self, completions, answer, **kwargs):
        rewards = []

        for completion, answer_i in zip(completions, answer):
            try:
                completion = "<think>" + completion

                is_formatted, reason_str = self.is_formatted(completion)
                if not is_formatted:
                    rewards.append(0.0)
                    continue
                
                answer_match = re.findall(r'<answer>\s*(.*?)\s*</answer>', completion, re.DOTALL)[-1].strip()
                reward = 1.0 if process_result_v1(answer_match, answer_i) else 0.1
                rewards.append(reward)

                if self.last_log_time is None:
                    self.last_log_time = time.time()
                if time.time() - self.last_log_time > 5:
                    self.last_log_time = time.time()
                    log_on_main(f"\n#########################\n{completion}\n-----\nFormat Explanation: {reason_str}\nTrue Answer: {answer_i}\n-----\nReward: {reward}\n#########################\n\n")

            except Exception as e:
                log_on_main(e)
                rewards.append(0.0)

        if hasattr(self, 'trainer') and hasattr(self.trainer, 'data_schedule'):
            if self.trainer.data_schedule == 'vrex':
                self.trainer._last_batch_rewards = rewards

        return rewards

    def gsm8k_reward_fn(self, completions, answer, **kwargs):
        """Reward function for gsm8k task"""
        rewards = []

        for completion, answer_i in zip(completions, answer):
            try:
                completion = "<think>" + completion

                is_formatted, reason_str = self.is_formatted(completion)
                if not is_formatted:
                    rewards.append(0.0)
                    continue

                answer_match = re.findall(r'<answer>\s*(.*?)\s*</answer>', completion, re.DOTALL)[-1].strip()
                reward = 1.0 if GSM8KRewardModel.is_correct(answer_match, answer_i) else 0.1
                rewards.append(reward)

                if self.last_log_time is None:
                    self.last_log_time = time.time()
                if time.time() - self.last_log_time > 5:
                    self.last_log_time = time.time()
                    log_on_main(f"\n#########################\n{completion}\n-----\nFormat Explanation: {reason_str}\nTrue Answer: {answer_i}\n-----\nReward: {reward}\n#########################\n\n")

            except Exception as e:
                log_on_main(e)
                rewards.append(0.0)

        if hasattr(self, 'trainer') and hasattr(self.trainer, 'data_schedule'):
            if self.trainer.data_schedule == 'vrex':
                self.trainer._last_batch_rewards = rewards

        return rewards
    
    def aqua_reward_fn(self, completions, answer, **kwargs):
        rewards = []

        for completion, answer_i in zip(completions, answer):
            try:
                completion = "<think>" + completion

                is_formatted, reason_str = self.is_formatted(completion)
                if not is_formatted:
                    rewards.append(0.0)
                    continue

                answer_match = re.findall(r'<answer>\s*(.*?)\s*</answer>', completion, re.DOTALL)[-1].strip()
                reward = 1.0 if answer_match==answer_i else 0.1
                rewards.append(reward)

                if self.last_log_time is None:
                    self.last_log_time = time.time()
                if time.time() - self.last_log_time > 5:
                    self.last_log_time = time.time()
                    log_on_main(f"\n#########################\n{completion}\n-----\nFormat Explanation: {reason_str}\nTrue Answer: {answer_i}\n-----\nReward: {reward}\n#########################\n\n")

            except Exception as e:
                log_on_main(e)
                rewards.append(0.0)

        if hasattr(self, 'trainer') and hasattr(self.trainer, 'data_schedule'):
            if self.trainer.data_schedule == 'vrex':
                self.trainer._last_batch_rewards = rewards

        return rewards


    def train(self):
        """Train a model using the specified algorithm with configurations from Hydra"""

        log_on_main('\n\n*****\ntrain\n*****\n\n')

        # Extract config values
        model_name = self.cfg.model.name
        output_model_name = self.cfg.output.run_name
        algorithm = self.cfg.algorithm.name

        # Load tokenizer & model
        model_config = self._get_model_config()
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path,
            trust_remote_code=model_config.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            torch_dtype=model_config.torch_dtype,
            trust_remote_code=model_config.trust_remote_code,
            attn_implementation=model_config.attn_implementation
        )
        peft_config = get_peft_config(model_config)

        # Prepare dataset
        dataset = self._prepare_dataset()
        dataset = dataset.map(lambda example: self._generate_prompt(tokenizer, example), remove_columns=dataset['train'].column_names)
        log_on_main(dataset)
        arithmetic_reward_fn = self.reward_functions[self.cfg.task.name]

        # Setup training arguments based on algorithm
        if "grpo" in algorithm:
            training_args = self._setup_grpo_training()
            trainer = CurriculumGRPOTrainer(
                model=model,
                reward_funcs=arithmetic_reward_fn,
                args=training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['test'],
                processing_class=tokenizer,
                peft_config=peft_config,
                num_tasks=len(self.cfg.task.training.data_files),
                total_iterations=training_args.max_steps,
                data_schedule=self.cfg.algorithm.training.curriculum_schedule,
                scheduler_params=self.cfg.algorithm.training.scheduler_params,
            )
            # Store trainer reference for reward function access
            self.trainer = trainer

        elif algorithm == "ppo":
            training_args = self._setup_ppo_training()
            trainer = PPOTrainer(
                model=model,
                ref_model=model_config.model_name_or_path,  # Same model as reference
                tokenizer=tokenizer,
                args=training_args,
                reward_fn=self._reward_fn,
                train_dataset=dataset['train'],
                eval_dataset=dataset['test'],
                peft_config=peft_config,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Train model
        trainer.train()
        trainer.save_model(training_args.output_dir)

        if self.cfg.algorithm.training.push_to_hub:
            trainer.push_to_hub(dataset_name='gsm8k-dataset')


    def inference(self):
        """Run inference using the trained model"""
        log_on_main('\n\n*****\ntest\n*****\n\n')

        output_dir = os.path.join("outputs", self.cfg.output.run_name)
        model = LLM(
            model=output_dir,
            trust_remote_code=self.cfg.model.trust_remote_code,
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=self.cfg.model.torch_dtype,
            gpu_memory_utilization=self.cfg.algorithm.training.vllm_gpu_memory_utilization,
            max_model_len=self.cfg.task.training.max_prompt_length+self.cfg.task.training.max_completion_length,
            seed=self.cfg.experiment.dataset_seed,
            task='generate'
        )

        tokenizer = model.get_tokenizer()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        sampling_params = SamplingParams(
            n=self.cfg.task.inference.n,
            temperature=self.cfg.task.inference.temperature,
            max_tokens=self.cfg.task.training.max_completion_length,
            min_tokens=1,
            seed=self.cfg.experiment.dataset_seed,
            stop=["</answer>"],
            include_stop_str_in_output=True
        )
        
        # Load and Preprocess Dataset  
        dataset = self._prepare_dataset()['test']
        dataset = dataset.map(lambda example: self._generate_prompt(tokenizer, example), remove_columns=dataset.column_names)
        dataset = dataset.remove_columns('sft')
        log_on_main(dataset)

        outputs = model.generate(dataset['prompt'], sampling_params)
        outputs = [
            completion_output.text
            for request_output in outputs
            for completion_output in request_output.outputs
        ]

        dataset = dataset.select([idx for idx in range(len(dataset['prompt'])) for _ in range(self.cfg.task.inference.n)])
        dataset = dataset.add_column('output', outputs)

        # Calcuate Rewards
        reward_fn = self.reward_functions[self.cfg.task.name]

        rewards = np.array(
            reward_fn(
                completions=dataset['output'],
                answer=dataset['answer']
            )
        )
        dataset = dataset.add_column('reward', rewards.tolist())
        dataset.to_json(os.path.join(output_dir, 'test_outputs.jsonl'))

        # Process Metrics
        results = dict()
        results['overall'] = {
            'avg_reward': rewards.mean().item(),
            'accuracy': (rewards > 0.5).mean().item(),
            'support': len(dataset)
        }
        
        for task_idx, data_dir in enumerate(self.cfg.task.inference.data_files):
            task_outputs = dataset.filter(lambda example: example['task']==task_idx)
            task_rewards = np.array(task_outputs['reward'])
            results[os.path.basename(os.path.normpath(data_dir))] = {
                'avg_reward': task_rewards.mean().item(),
                'accuracy': (task_rewards > 0.5).mean().item(),
                'support': len(task_rewards)
            }

        log_on_main(json.dumps(results, indent=4))
        with open(os.path.join(output_dir, 'test_results.json'), "w") as f:
            json.dump(results, f, indent=4)

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

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
        for _ in range(self.num_generations):
          if self.model_type == "completion":   
              outputs.append(self.base_model.generate(inputs,
                                            hide_input=True,
                                            do_sample=True,
                                            skip_special_tokens=False,
                                            temperature=0.0).text) 
        outputs = [list(group) for group in zip(*outputs)]
        return outputs  


def occupy_gpu_memory(gb=75, device="cuda:0"):
    """
    Allocates a tensor on the specified GPU that occupies approximately `gb` GB of memory.
    The tensor remains allocated indefinitely (until the process is terminated).
    """
    # Calculate the target memory in bytes.
    target_bytes = gb * 1024 ** 3
    # For float32, each element takes 4 bytes.
    num_elements = target_bytes // 4
    torch.cuda.empty_cache()
    print(f"Allocating a tensor with {num_elements} float32 elements (~{gb}GB) on {device}.")

    try:
        # Allocate the tensor on the specified device.
        tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
        tensor.fill_(0)
        print(f"Successfully allocated ~{gb}GB on {device}. Holding memory indefinitely...")
    except RuntimeError as e:
        print("Failed to allocate memory. Your GPU may not have enough free memory.")
        raise e

    # Hold the memory indefinitely.
    while True:
        print("Holding memory...")
        time.sleep(60)


class CodeTrainer(BaseTrainer):
    """Class for training and inference on code models"""


    def _prepare_dataset(self):
        """Prepare dataset for training"""
        all_samples = []
        for task_idx, file in enumerate(self.cfg.task.data_files):
            file_dataset = load_dataset('json', data_files=file)['train']
            # Annotate with difficulty
            file_dataset = file_dataset.add_column("task", [task_idx] * len(file_dataset))
            all_samples.extend(file_dataset)
        dataset = Dataset.from_list(all_samples)
        dataset = dataset.shuffle(seed=self.cfg.experiment.dataset_seed)
        return dataset

    def _get_info(self, example):
            question = f"""
            {example.get('description')}
            Input format:
            {example.get('input_format')}
            Output format:
            {example.get('output_format')}
            Examples:
            {example.get('examples')}
            Notes:
            {example.get('note')}
            """
            
            verification = example.get('official_tests')
            
            id = example.get('id')
            
            prompt = f"Solve the following coding problem\n{question}\n\n Show your work in <think> </think> tags. And return the final code in <code> </code> tags. Read from the stdin and write to the stdout. For example <code> ```python\nprint(1)\n``` </code>."
            
            return prompt, verification, id

    class PromptOutput(TypedDict):
        prompt: str
        question: str
        test_cases: List
        question_id: str
        task: int

    def _generate_prompt(self, tokenizer, example) -> PromptOutput:
        """Generate prompt for the coding model"""
        # Extract target and numbers from the example


        # data = example.get("reward_model", {}).get("ground_truth", {})
        # target = data.get("target")
        # numbers = data.get("numbers")
        # # expression = data.get("expression") # e.g., (((76 - 80) - 28) + 43), (((65 * 12) + 60) / 28)
        question, verification, id = self._get_info(example)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful python coding assistant. You first think about the reasoning process in the mind and then provides the user with the code to solve the problem.\n"
            },
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>"
            }
        ]

        # tokens = tokenizer.apply_chat_template(messages, tokenize=True, continue_final_message=True)
        # print("Number of tokens in the prompt:", len(tokens))
        return {
            "prompt": tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True),
            "question": question,
            "test_cases": verification,
            "question_id": id,
            "task" : example["task"]
        }

    def _validate_coding_response_format(self, response: str):
        """Validate the coding response format"""
        # Remove leading/trailing whitespace
        response = response.strip()

        # Must contain <think> and </think> tags
        if "<think>" not in response or "</think>" not in response:
            print('Response does not contain think tags')
            return False

        # Must contain <code> and </code> tags
        if "<code>" not in response or "</code>" not in response:
            print('Response does not contain code tags')
            return False
        
        if not any(sub in response for sub in ("input()", "sys.stdin")):
            print("Response doesn't read from stdin")
            return False
            

        # Check that tags are in correct order
        think_open = response.find("<think>")
        think_close = response.find("</think>")
        code_open = response.find("<code>")
        code_close = response.find("</code>")

        if think_close < think_open or code_close < code_open:
            return False

        if code_open < think_close:
            return False

        if (response.count("<think>")   != 1 or
                response.count("</think>")  != 1 or
                response.count("<code>")    != 1 or
                response.count("</code>")   != 1):
                return False

        return True

    def _coding_reward_fn(self, completions, question, test_cases, question_id, **kwargs):
        """Reward function for coding task"""
        rewards = []
        for completion, test_case, id in zip(completions, test_cases, question_id):
            try:
                print('#########################')
                completion = "<think>" + completion
                print(completion)

                if not self._validate_coding_response_format(completion):
                    print('Response Format Error')
                    rewards.append(0.0)  # Penalty to avoid format errors
                    continue

                reward_model = CodingRewardModel(test_case) #TODO Set this to be test_cases
                reward = reward_model.compute_score(completion)
                rewards.append(reward)
                print('-----')
                print(reward)
                print('-----')
                print('#########################')
            except Exception as e:
                print(e)
                rewards.append(0.0)

        # Update variance regularized scheduler if we're in training mode
        # and task IDs are available
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'data_schedule'):
            if self.trainer.data_schedule == 'vrex':
                # Store rewards in trainer for later use
                self.trainer._last_batch_rewards = rewards

        return rewards

    def train(self):
        """Train a model using the specified algorithm with configurations from Hydra"""
        # Extract config values
        model_name = self.cfg.model.name
        output_model_name = self.cfg.output.run_name
        algorithm = self.cfg.algorithm.name

        # Load tokenizer and model
        model_config = self._get_model_config()
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path,
            trust_remote_code=model_config.trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            torch_dtype=model_config.torch_dtype,
            trust_remote_code=model_config.trust_remote_code,
            attn_implementation=model_config.attn_implementation
        )
        peft_config = get_peft_config(model_config)

        # Prepare dataset
        # train_dataset, test_dataset = self._prepare_dataset(self.cfg.task.data_files)
        # train_dataset = train_dataset.map(lambda example: self._generate_prompt(tokenizer, example))
        # test_dataset = test_dataset.map(lambda example: self._generate_prompt(tokenizer, example))

        dataset = self._prepare_dataset()
        dataset = dataset.map(lambda example: self._generate_prompt(tokenizer, example), remove_columns=dataset.column_names)
        log_on_main(dataset)
        # Split dataset
        # train_test_split = dataset.train_test_split(test_size=self.cfg.task.test_size)
        # train_dataset = train_test_split["train"]
        # test_dataset = train_test_split["test"]

        # Setup training arguments based on algorithm

        if "grpo" in algorithm:
            training_args = self._setup_grpo_training()
            trainer = CurriculumGRPOTrainer(
                model=model,
                reward_funcs=self._coding_reward_fn,
                args=training_args,
                train_dataset=dataset,
                processing_class=tokenizer,
                peft_config=peft_config,
                num_tasks=len(self.cfg.task.data_files),
                total_iterations=training_args.max_steps,
                data_schedule=self.cfg.algorithm.training.curriculum_schedule,
                scheduler_params=self.cfg.algorithm.training.scheduler_params,
            )
            # Store trainer reference for reward function access
            self.trainer = trainer
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Train model
        trainer.train()
        trainer.save_model(training_args.output_dir)

        if self.cfg.algorithm.training.push_to_hub:
            trainer.push_to_hub(dataset_name='coding-dataset')

    def inference(self):
        """Run inference using the trained model"""
        # Extract config values
        model_checkpoint = self.cfg.task.inference.checkpoint
        sc_num = self.cfg.task.inference.sc_num

        # Generate checkpoint path
        model_dir = self._get_checkpoint_path(model_checkpoint, self.cfg.model.trim)

        # Load test dataset
        _, test_dataset = self._prepare_dataset([self.cfg.task.test_file])

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=self.cfg.model.trust_remote_code
        )

        # Custom model for inference
        model = HFModel(
            model_pth=model_dir,
            tokenizer_pth=model_dir,
            max_new_tokens=self.cfg.task.inference.max_new_tokens
        )

        # Run inference on test dataset
        correct = 0
        rewards = 0
        total = 0
        results = []

        for example in tqdm(test_dataset):
            # Generate prompt
            prompt_data = self._generate_prompt(tokenizer, example)
            prompt = prompt_data["prompt"]

            # Generate responses
            outputs = []
            for _ in range(sc_num):
                output = model.generate([prompt], do_sample=True, temperature=0.0, verbose=False, skip_special_tokens=False).text[0]
                outputs.append(output)

            # Evaluate responses
            for output in outputs:
                # Prepare ground truth for scoring
                test_cases = prompt_data.get("test_cases", [])
                
                # Use the CodingRewardModel for evaluation
                reward_model = CodingRewardModel(test_cases=test_cases)

                # Calculate score
                score = reward_model.compute_score(output)

                # Extract solution
                solution = reward_model.extract_solution(output)

                # Record results
                results.append({
                    "prompt": prompt,
                    "output": output,
                    "solution": solution,
                    "test_cases": test_cases,
                    "score": score
                })

                rewards += score
                if score > 0.5:  # We are doing binary scoring so over 0.5 will always be correct
                    correct += 1
                total += 1

        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
        rewards /= total if total > 0 else 0
        print(f'Accuracy: {accuracy}, Rewards: {rewards}')

        # Save results to output directory
        evaluation_results = {
            "accuracy": accuracy,
            "rewards": rewards,
            "model_checkpoint": model_checkpoint,
            "sc_num": sc_num,
            "detailed_results": results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # with open(os.path.join(model_dir, "inference_results.json"), "w") as f:
        #     json.dump(evaluation_results, f, indent=2)

        return accuracy

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Main entry point for training and inference with Hydra configuration"""
    print(OmegaConf.to_yaml(cfg))

    # Select the appropriate trainer based on the task
    task = cfg.task.name
    if "blocksworld" in task:
        trainer = BlocksWorldTrainer(cfg)
    elif "countdown" in task:
        trainer = CountdownTrainer(cfg)
    elif any(x in task for x in ["gsm8k", "math", "aqua"]):
        trainer = ArithmeticTrainer(cfg)
    elif "code" in task:
        trainer = CodeTrainer(cfg)
    else:
        raise ValueError(f"Unknown task: {task}. Choose either 'blocksworld', 'countdown', or 'gsm8k'")

    # Check which mode to run
    if cfg.mode == "train":
        trainer.train()
    elif cfg.mode == "inference":
        trainer.inference()
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Choose either 'train' or 'inference'")

    # Optional: Occupy GPU memory after training (useful for server environments)
    if cfg.get("occupy_gpu_memory", False):
        occupy_gpu_memory(gb=cfg.occupy_gpu_memory_gb, device=cfg.gpu_device)


if __name__ == "__main__":
    main()