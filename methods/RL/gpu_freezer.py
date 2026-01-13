import time
import torch
def occupy_gpu_memory(gb=75, device="cuda:0"):
    """
    Allocates a tensor on the specified GPU that occupies approximately `gb` GB of memory.
    The tensor remains allocated indefinitely (until the process is terminated).

    Parameters:
      - gb (int): Target memory in gigabytes (default: 75).
      - device (str): The CUDA device (e.g. "cuda:0") to allocate on.
      
    Note: If you set CUDA_VISIBLE_DEVICES=7,2 then "cuda:0" is physical GPU 7.
    """
    # Calculate the target memory in bytes.
    target_bytes = gb * 1024**3  
    # For float32, each element takes 4 bytes.
    num_elements = target_bytes // 4

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

occupy_gpu_memory(gb=75, device="cuda:5")