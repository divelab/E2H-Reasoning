import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List
from dataclasses import dataclass, field
from peft import get_peft_model, LoraConfig, TaskType
import time
import random
import numpy as np
import math
@dataclass
class ModelConfig:
    model_name_or_path: str
    torch_dtype: str = "float16"
    attn_implementation: Optional[str] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    lora_target_modules: Optional[List[str]] = field(default_factory=list)
    lora_task_type: str = "CAUSAL_LM"

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




def load_model(config: ModelConfig):
    dtype = getattr(torch, config.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        use_cache=False,
        
    )
    
    if config.attn_implementation:
        model.config.attn_implementation = config.attn_implementation
    
    if config.lora_r and config.lora_alpha and config.lora_target_modules:
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout or 0.0,
        )
        
        model = get_peft_model(model, peft_cfg)
    
    model.eval()
    return tokenizer, model

def avg_logprob(prefix: str, continuation: str, tokenizer, model) -> float:
    x_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device) # X
    y_ids = tokenizer(continuation, return_tensors="pt").input_ids.to(model.device) # Y
    inputs = torch.cat([x_ids, y_ids], dim=1)
    print(f'Total Length - {inputs.shape}')
    with torch.no_grad():
        logits = model(inputs).logits
    # torch.cuda.synchronize()
    log_probs = F.log_softmax(logits, dim=-1) # P(Y|X)
    start = x_ids.shape[1]
    token_logps = [
        log_probs[0, start + i, y_ids[0, i]].item()
        for i in range(y_ids.shape[1])
    ]
    print(len(token_logps))
    avg_lp = sum(token_logps) / len(token_logps)
    
    total_token_logps = [
        log_probs[0, i, inputs[0, i]].item()
        for i in range(1, inputs.shape[1])
    ]
    avg_lp_all = sum(total_token_logps) / len(total_token_logps)
    print(avg_lp_all)
    return avg_lp

def top_k_next_tokens(prefix_ids, y_ids, model, parent_score, k=10, chunk_size=512, max_improvements=1000):
    vocab_size = model.config.vocab_size
    best = []
    improvements_seen = 0

    for start in range(0, vocab_size, chunk_size):
        if improvements_seen >= max_improvements:
            print(f"Stopping early after {improvements_seen} improvements")
            break

        end = min(start + chunk_size, vocab_size)
        t0 = time.perf_counter()

        cand_ids = torch.arange(start, end, device=model.device).unsqueeze(1)
        batch_prefix = prefix_ids.repeat(end - start, 1)
        batch_prefix[:, -1] = cand_ids.squeeze()
        batch_y = y_ids.repeat(end - start, 1)
        inputs = torch.cat([batch_prefix, batch_y], dim=1)

        with torch.no_grad():
            logits = model(inputs).logits

        log_probs = F.log_softmax(logits, dim=-1)
        seq_start = prefix_ids.shape[1]
        avg_lp = torch.stack([
            log_probs[:, seq_start + i, batch_y[0, i]]
            for i in range(batch_y.shape[1])
        ], dim=1).mean(dim=1)

        # Count improvements in this chunk
        chunk_improvements = (avg_lp > parent_score).sum().item()
        improvements_seen += chunk_improvements

        vals, ids = torch.topk(avg_lp, k)
        for score, tok in zip(vals.tolist(), (ids + start).tolist()):
            if score > parent_score:
                best.append((score, tok))

        best = sorted(best, reverse=True)

        elapsed = time.perf_counter() - t0
        print(f"Chunk [{start}:{end}] → {elapsed:.3f}s | chunk_impr={chunk_improvements} | total_impr={improvements_seen}")

    return [(tok, score) for score, tok in best]

def iterative_search(tokenizer, model, X, Y, steps=1, topk=10, beam_size=None, max_improvements=100):
    """
    Greedy or beam search to append up to `steps` tokens.
    Only keeps expansions that strictly increase avg_logprob(Y|prefix).
    Returns a list of (appended_tokens, score).
    """
#     base_x = """Using the numbers [19,36,55,7], create an equation that equals 65.\n<think>\nWe need to find an equation using the numbers 19, 36, 55, and 7 exactly once, with basic arithmetic operations, that equals 65.
# One possible combination is 55 + 36 - 19 + 7. Let's check: 55 + 36 = 91, 91 - 19 = 72, and 72 + 7 = 79. However, this doesn't equal 65.
# Another combination is 55 + 36 + 7 - 19. Let's check: 55 + 36 = 91, 91 + 7 = 98, and 98 - 19 = 79. This also doesn't equal 65.
# After trying different combinations, I found that 55 + 36 + 7 - 19 = 79, which is close but not equal to 65.
# Finally, I found that 55 + 36 - 19 + 7 = 79, which is still not equal to 65.
# However, if we try 55 + 36 - 7 + 19, we get 55 + 36 = 91, 91 - 7 = 84, and 84 + 19 = 103, which is not equal to 65.
# After further trial and error, I found that 55 + 36 + 7 - 19 = 79, which is still not equal to 65.
# However, if we try 55 + 36 - 7 - 19, we get 55 + 36 = 91, 91 - 7 = 84, and 84 - 19 = 65. This equals 65.\n</think>
# """
    # base_x = f"Using the numbers [19,36,55,7], create an equation that equals 65."
    
    base_score = avg_logprob(X, Y, tokenizer, model)
    print(base_score)
    x_ids = tokenizer(X, return_tensors="pt").input_ids.to(model.device)
    y_ids = tokenizer(Y, return_tensors="pt").input_ids.to(model.device)
    beams = [(x_ids, base_score)]

    for step in range(steps):
        new_beams = []
        step_start = time.perf_counter()
        for prefix_ids, parent_score in beams:
            print('Parent Score - ', parent_score)
            for tok, score in top_k_next_tokens(prefix_ids, y_ids, model, parent_score, k=topk, max_improvements=max_improvements):
                if score <= parent_score:
                    continue
                new_prefix = torch.cat([prefix_ids, torch.tensor([[tok]], device=model.device)], dim=1)
                full_seq = tokenizer.decode(new_prefix[0])
                print(full_seq, score)
                new_beams.append((new_prefix, score))
        if not new_beams:
            break
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size] if beam_size else [new_beams[0]]
        print(f"Step {step} time: {time.perf_counter() - step_start:.3f}s")

    orig_len = x_ids.shape[1]
    results = []
    for prefix_ids, score in beams:
        full_seq = tokenizer.decode(prefix_ids[0])
        appended = tokenizer.decode(prefix_ids[0][orig_len:])
        results.append((full_seq, appended, score))
    del x_ids
    del y_ids
    return results


def compute_avg_logprob(cfg: ModelConfig, prefix: str, continuation: str, tokenizer=None, model=None):
    # tokenizer, model = load_model(cfg)
    score = avg_logprob(prefix, continuation, tokenizer, model)
    print(f"Avg log‑prob = {score:.4f}")
    return score

if __name__ == "__main__":
    config = ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-3B",
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
    )
    
    
    
    # compute_avg_logprob(config, f"Using the numbers [19,36,55,7], create an equation that equals 65.{reasoning_text}", "<answer> 55 + 36 - 7 - 19 </answer>")
    
    # 
    prefix = "Using the numbers [19,36,55,7], create an equation that equals 65. "
    y = "the equation is 55 + 36 - 7 - 19"
    tokenizer, model = load_model(config)
    compute_avg_logprob(config, prefix, y, tokenizer=tokenizer,model=model) # Show your work in the <think> </think> tags.
    prefix = """Using the numbers [19,36,55,7], create an equation that equals 65. We need to find an equation using the numbers 19, 36, 55, and 7 exactly once, with basic arithmetic operations, that equals 65. 
One possible combination is 55 + 36 - 19 + 7. Let's check: 55 + 36 = 91, 91 - 19 = 72, and 72 + 7 = 79. However, this doesn't equal 65.
Another combination is 55 + 36 + 7 - 19. Let's check: 55 + 36 = 91, 91 + 7 = 98, and 98 - 19 = 79. This also doesn't equal 65.
After trying different combinations, I found that 55 + 36 + 7 - 19 = 79, which is close but not equal to 65.
Finally, I found that 55 + 36 - 19 + 7 = 79, which is still not equal to 65.
However, if we try 55 + 36 - 7 + 19, we get 55 + 36 = 91, 91 - 7 = 84, and 84 + 19 = 103, which is not equal to 65.
After further trial and error, I found that 55 + 36 + 7 - 19 = 79, which is still not equal to 65.
However, if we try 55 + 36 - 7 - 19, we get 55 + 36 = 91, 91 - 7 = 84, and 84 - 19 = 65. This equals 65. Therefore 
"""
    compute_avg_logprob(config, prefix, y,tokenizer=tokenizer,model=model)
    prefix = "Using the numbers [19,36,55,7], create an equation that equals 65."
    compute_avg_logprob(config, prefix, y,tokenizer=tokenizer,model=model)
    compute_avg_logprob(config, prefix, y,tokenizer=tokenizer,model=model)
    compute_avg_logprob(config, prefix, y,tokenizer=tokenizer,model=model)
    
    print("Greedy result:", iterative_search(tokenizer, model, prefix, y, steps=20, topk=10, max_improvements=20))
    
    # print("Greedy result:")
    # results_greedy = iterative_search(tokenizer, model, prefix, y, steps=1, topk=10, max_improvements=100)
    # for full_seq, appended, score in results_greedy:
    #     print(f"\nScore {score:.4f}")
    #     print(f"Full sequence: {full_seq}")
    #     print(f"Appended tokens: {appended}")

    # print("\nBeam search (beam size=3) result:")
    
    results_beam = iterative_search(tokenizer, model, prefix, y, steps=1, topk=10, beam_size=5, max_improvements=100)
    for full_seq, appended, score in results_beam:
        print(f"\nScore {score:.4f}")
        print(f"Full sequence: {full_seq}")
        print(f"Appended tokens: {appended}")
    