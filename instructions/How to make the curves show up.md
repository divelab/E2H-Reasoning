GRPOTrainer.log() is **only a staging call** – it puts the key/value pairs in the trainer’s internal

self.stats dictionary.

Nothing is actually sent to Weights & Biases / TensorBoard until the trainer later calls

trainer.log_stats(...).

Inside log_stats the contents of self.stats are flushed to the real logger and then the

dictionary is emptied:

```
def log_stats(self, stats, step=None, **kwargs):
    self.stats.update(stats)          # merge new values
    ...
    if self.config.log_with == "wandb":
        wandb.log(self.stats, step=step)
    ...
    self.stats = {}                   # <- buffer is cleared here
```

If you do your extra trainer.log(...) **after** the trainer has already called

log_stats() for the current step, the buffer is immediately cleared, so the values never leave

your process and no curve appears in the UI.





### **How to make the curves show up**





1. **Queue them before the trainer’s own flush**



```
# build one dict first
extras = {}

for i, prob in current_probs.items():
    extras[f"vrex/task_{i}_sampling_prob"] = prob
for i, w in enumerate(group_weights):
    extras[f"vrex/task_{i}_groupdro_weight"] = w
for i, cnt in state["task_counts"].items():
    extras[f"vrex/task_{i}_sample_frequency"] = cnt / max(total_counts, 1)
for i, mastered in state["task_mastery"].items():
    extras[f"vrex/task_{i}_mastery"] = float(mastered)

trainer.log(extras)   # call this **before** the internal log_stats() happens
```



1. 
2. **—or— flush yourself**



```
trainer.log_stats(extras, step=trainer.iteration)
```



1. 
2. **—or— bypass the trainer and log directly**



```
if trainer.accelerator.is_main_process:
    wandb.log(extras, step=trainer.iteration)
```







### **Where to look in the UI**





The slash (/) in keys such as vrex/task_0_sampling_prob makes TensorBoard and W&B put the

curves under the *vrex* group.

Expand that group in the left-hand panel (TensorBoard) or search for *vrex/* in W&B and the

signals will be there.



In short, your values weren’t visible because they were added to the buffer **after** it had

already been flushed and cleared. Log them earlier, flush them yourself, or log directly, and the

curves will appear.