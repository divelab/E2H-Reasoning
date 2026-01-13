from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
# Sample values for demonstration
numbers = "1, 2, 3, 4"
target = "10"
icl_example = ""
# Define the chat template
r1_prefix = [{
                "role": "system",
                "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
        },
        { 
                "role": "user",
                "content": f"I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nHere is the format of the actions: \n\npick up the [block_name] block # for example: pick up the blue block\nunstack the [block_name] block from on top of the [another_block_name] block # for example: unstack the orange block from on top of the black block\nput down the [block_name] block # for example put down the red block\nstack the [block_name] block on top of the [another_block_name] block # for example: stack the yellow block on top of the red block \n\n{icl_example}\n\n[Problem]\nHere is the initial state of the blocks: the cyan block is clear, the ruby block is clear, the hand is empty, the cyan block is on top of the emerald block, the emerald block is on top of the magenta block, the magenta block is on the table and the ruby block is on the table\n\nHere is the goal state of the blocks: the cyan block is on top of the ruby block and the emerald block is on top of the magenta block.\nShow your work in the <think> </think> tags. Return the final sequence of actions as the plan in the <plan> </plan> tags.\n" # , for example: <plan>\npick up the blue block\nstack the blue block on top of the yellow block\nunstack the orange block from on top of the black block\nstack the orange block on top of the red block</plan>
        },
        {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>"
        }
    ]

# Generate the full prompt text from the chat template
full_prompt = tokenizer.apply_chat_template(
    r1_prefix, tokenize=False, continue_final_message=True
)

# Tokenize the full prompt
token_ids = tokenizer.encode(full_prompt)
print(len(token_ids))
# Truncate to 256 tokens
truncated_tokens = token_ids[-256:]
truncated_prompt = tokenizer.decode(truncated_tokens)

print("Truncated prompt (max 256 tokens):\n")
print(truncated_prompt)
