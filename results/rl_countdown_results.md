# RL Countdown Task Results

## Experiment Overview
- **Date**: May 10-25, 2025
- **Model**: Qwen2.5-1.5B-Instruct
- **Task**: Countdown (Arithmetic Expression Generation)
- **Method**: GRPO (Group Relative Policy Optimization)

## Training Results

| Model Configuration | Task | Algorithm | Curriculum Schedule | Training Steps | Final Loss | Final Reward | Training Time | WandB Run |
|-------------------|------|-----------|-------------------|----------------|------------|--------------|---------------|-----------|
| Qwen2.5-1.5B-Instruct | countdown345 | GRPO | balanced (μ=0.5, σ=0.5) | 1600 | 0.0444 | 0.4022 | 2:13:31 | [o1bh26qh](https://wandb.ai/dive-ci/Sys2Bench/runs/o1bh26qh) |

## Inference Results

| Window | Model/Checkpoint | Accuracy (%) | Accuracy (Raw) | Reward Score | Notes |
|--------|-----------------|--------------|----------------|--------------|-------|
| 0 | Unknown | **9.47%** | 0.0947265625 | 0.1831 | Lowest performance |
| 1 | Unknown | **62.50%** | 0.625 | 0.6325 | Good performance |
| 2 | Unknown | **79.20%** | 0.7919921875 | 0.8117 | 🏆 **Best Result** |
| 3 | Unknown | **30.08%** | 0.30078125 | 0.3697 | Below average |
| 4 | Unknown | **21.88%** | 0.21875 | 0.2967 | Poor performance |

## Key Observations

1. **Performance Variance**: Results show high variance across different windows/checkpoints (9.47% - 79.20%)
2. **Best Performance**: Window 2 achieved the highest accuracy at 79.20%
3. **Correlation**: Strong correlation between accuracy and reward scores
4. **Average Performance**: Mean accuracy across all tests ≈ 40.6%

## Technical Details

- **Server**: dive7.engr.tamu.edu
- **GPUs Used**: NVIDIA A100 80GB PCIe
- **Framework**: PyTorch with Accelerate/DeepSpeed
- **Inference Batch Size**: 32 (based on progress indicators)

## Notes

- All inference runs completed successfully with "Testing batches: 100%"
- Some runs show different processing speeds (e.g., Window 2 had speeds up to 19,599 toks/s)
- Results collected from tmux session: `exps_sys2bench_1746905497`

---
*Generated on: May 25, 2025*