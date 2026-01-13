If you're working with TRL trainers (like PPOTrainer or SFTTrainer) and want to log custom metrics during training, here's how you can generally achieve that, particularly when using a logging backend like Weights & Biases (W&B) or TensorBoard: 
1. Utilize a Supported Experiment Tracker:
Specify report_to in your TRL configuration: When initializing your TRL trainer configuration (e.g., PPOConfig), specify the experiment tracker you want to use by setting the report_to parameter. For example, to use Weights & Biases:
python
from trl import PPOConfig

training_args = PPOConfig(
    ...,
    report_to="wandb"  # or "tensorboard"
)
Configure the logging directory for TensorBoard (if applicable): If using TensorBoard, you may need to specify the logging directory using project_kwargs={"logging_dir": PATH_TO_LOGS} within your configuration object. 
2. Leverage Callbacks for Custom Logging:
Subclass WandbCallback (or a similar callback for your chosen logger): If you need to log metrics beyond the default TRL logs, you can create a custom callback that subclasses the appropriate logging callback for your chosen tracker (e.g., WandbCallback for W&B).
Override methods to log extra metrics: Within your custom callback, you can override methods like on_log, on_step_end, or on_epoch_end to log the desired metrics.
Access the Trainer and its state: Your custom callback will have access to the Trainer object, allowing you to access training state and potentially compute metrics as needed.
Log data using the logger's API: Within the overridden methods, use the logger's API (e.g., wandb.log()) to log your custom metrics. 
3. Example (WandbCallback):
Here's a conceptual example using a custom WandbCallback to log additional metrics during training: 
python
from transformers import Trainer
from transformers.integrations import WandbCallback
import wandb # Assuming you have wandb installed

class CustomWandbCallback(WandbCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs, **kwargs)

        if logs is not None:
            # Example: Log a custom metric calculated during a training step
            if 'my_custom_metric' in logs:
                wandb.log({'my_custom_metric': logs['my_custom_metric']}, step=state.global_step)

            # You can also log other relevant data here, like evaluation samples

# Instantiate your Trainer and callback
trainer = Trainer(...) # Your TRL Trainer instance
evals_callback = CustomWandbCallback(trainer=trainer, ...) # Initialize your custom callback

# Add the callback to the Trainer
trainer.add_callback(evals_callback)

# Start training
trainer.train()
Important Notes:
Refer to the documentation: Consult the documentation for your specific TRL trainer and the experiment tracker you're using for the most up-to-date and precise instructions on custom logging.
Metrics per epoch vs. per step: If you want to log metrics per epoch, ensure evaluation_strategy="epoch" is set in your TrainingArguments.
compute_metrics function: You can also pass a compute_metrics function to the trainer for custom evaluation metrics, which can be useful for validating on metrics you need.
Logging directory for TensorBoard: If using TensorBoard, remember to specify the logging directory in your configuration to ensure metrics are saved correctly. 
Logging - Hugging Face
As reinforcement learning algorithms are historically challenging to debug, it's important to pay careful attention to logging. By default, TRL trainers like PP...
favicon
Hugging Face
Hugging Face Transformers | Weights & Biases Documentation
Jun 5, 2025 — How do I log and view evaluation samples during training. Logging to W&B via the Transformers Trainer is taken care of by the WandbCallback in the Transformers ...
favicon
Weights & Biases Documentation

trl/docs/source/logging.md at main · huggingface/trl - GitHub
As reinforcement learning algorithms are historically challenging to debug, it's important to pay careful attention to logging. By default, the TRL [ PPOTrainer...
favicon
GitHub
Show all