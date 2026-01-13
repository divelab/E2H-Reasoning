import torch

def bfs_pronto_extractor(algo_output):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # to make sure the plan is saved before evaluation in multi-process setting
    try:
        answer = "\n".join(algo_output.terminal_node.state[2::2])
        answer = answer.replace("So ", "")
        answer = answer.replace("So, ", "")
        return answer

    except Exception as e:
        print("Error in output extraction,", e)
        return ""

def dfs_bw_extractor(algo_output):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # to make sure the plan is saved before evaluation in multi-process setting
    try:
        answer = "\n".join(algo_output.terminal_state[2::2])
        answer = answer.replace("So ", "")
        return answer

    except Exception as e:
        print("Error in output extraction,", e)
        return ""