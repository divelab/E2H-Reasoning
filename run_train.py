# run_train.py
import os
import sys
from accelerate.commands.launch import launch_command, main

if __name__ == "__main__":
    # ---- 1a. Environment variables exactly as in your shell one-liner
    # print('start path', sys.path)
    os.environ["WANDB_PROJECT"] = "Sys2Bench"
    os.environ["ROOT_PATH"]     = "/data/shurui.gui/Projects/gateway/Sys2Bench"
    # Fix VLLM compatibility issue - force V0 engine
    # os.environ["VLLM_USE_V1"] = "0"
    sys.argv = sys.argv[1:]
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[0].replace('cuda_visible_devices=', '')
    #
    # # ---- 1b. Recreate the CLI arguments list
    # sys.argv = [
    #     # "accelerate",                      # dummy entry-point
    #     "launch",
    #     "--num_processes", "1",
    #     "--main_process_port", "29763",
    #     "--config_file", "methods/RL/deep_speed.yaml",
    #     #
    #     "methods/RL/main.py",
    #     "mode=train",
    #     "task=countdown2345",
    #     "task.train_size=1000",
    #     "algorithm=grpo",
    #     "algorithm.training.curriculum_schedule=variance_regularized",
    #     "model=qwen15",
    #     "algorithm.training.per_device_train_batch_size=2",
    #     "algorithm.training.max_steps=1600",
    #     "algorithm.training.vllm_gpu_memory_utilization=0.6",
    #     "+algorithm.training.scheduler_params.beta=0.6",
    #     "+algorithm.training.scheduler_params.progression_bias=0.2",
    #     "+algorithm.training.scheduler_params.performance_threshold=0.7",
    # ]

    # ---- 1c. Let Accelerate’s normal entry-point handle everything
    sys.argv = ['launch'] + sys.argv[1:]
    # print(sys.argv)
    main()