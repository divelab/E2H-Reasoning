"""
Patch to add variance regularized scheduler to main.py

Add this import at the top of main.py:
from variance_regularized_scheduler import _variance_regularized_schedule, update_variance_regularized_performance, reset_variance_regularized_state

Then modify the schedule_funcs dictionary in TaskSampler.__init__ (around line 180):

        self.schedule_funcs = {
            'balanced': self._balanced_schedule,
            'cosine': self._cosine_schedule,
            'gaussian': partial(self._gaussian_schedule, **scheduler_params),
            'classic': self._step_schedule,
            'variance_regularized': partial(_variance_regularized_schedule, **scheduler_params)
        }

And add this method to the CurriculumGRPOTrainer class to update performance:

    def compute_rewards(self, *args, **kwargs):
        # Call parent method
        rewards = super().compute_rewards(*args, **kwargs)
        
        # Update variance regularized scheduler if using it
        if hasattr(self, 'data_schedule') and self.data_schedule == 'variance_regularized':
            # Extract task IDs and performances from the current batch
            # This assumes the batch has 'task' field and rewards are computed
            if hasattr(self, 'train_dataset') and 'task' in self.train_dataset.column_names:
                batch_indices = self._get_train_sampler().batch_indices  # Need to track this
                task_ids = [self.train_dataset[idx]['task'] for idx in batch_indices]
                performances = rewards.cpu().numpy().tolist()
                update_variance_regularized_performance(task_ids, performances)
        
        return rewards
"""

# Example usage configuration
example_config = """
# Example configuration for training with variance regularized scheduler:

WANDB_PROJECT=Sys2Bench ROOT_PATH=/data/shurui.gui/Projects/Sys2Bench CUDA_VISIBLE_DEVICES=0,1 accelerate launch \\
    --num_processes 1 \\
    --main_process_port=29850 \\
    --config_file methods/RL/deep_speed.yaml \\
    methods/RL/main.py \\
    mode=train \\
    task=countdown6 \\
    algorithm=grpo \\
    algorithm.training.curriculum_schedule=variance_regularized \\
    model=qwen15 \\
    algorithm.training.per_device_train_batch_size=2 \\
    algorithm.training.scheduler_params.min_prob=0.1 \\
    algorithm.training.scheduler_params.temperature=1.0 \\
    algorithm.training.scheduler_params.beta=0.5 \\
    algorithm.training.scheduler_params.warmup_steps=100 \\
    algorithm.training.scheduler_params.vrex_penalty_weight=1.0 \\
    algorithm.training.scheduler_params.groupdro_alpha=0.01 \\
    algorithm.training.max_steps=1600
"""

print(example_config)