import re
import torch
import numpy as np
from typing import Union
from trl import GRPOTrainer
from functools import partial
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
            scheduler_params=self.scheduler_params
        )

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
                # TODO: Resample (self.args.generation_batch_size // self.num_generations) New prompts.
                # TODO: Inputs will be a list of size 16. (2 new prompts, repeated 8 times).
                # TODO: Make resample func in sampler that directly gives these prompts
                result = self._generate_and_score_completions(inputs)
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
