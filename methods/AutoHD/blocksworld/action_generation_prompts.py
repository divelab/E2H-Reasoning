from utils import make_actions
# In Context Examples
state_empty = ['the blue block is clear, the orange block is clear, the hand is empty, the blue block is on top of the red block, the orange block is on the table, and the red block is on the table.',
        #  'the orange block is clear, the yellow block is clear, the hand is empty, the orange block is on top of the red block, the yellow block is on top of the blue block, the blue block is on the table, and the red block is on the table.',
         'the blue block is clear, the hand is empty, the red block is on top of the yellow block, the blue block is on top of the red block, the yellow block is on top of the orange block and the orange block is on the table.',
        #  'the green block is clear, the orange block is clear, the red block is clear, the yellow block is clear, the hand is empty, the green block is on the table, the orange block is on the table, the red block is on the table, and the yellow block is on the table.',
         'the green block is clear, the orange block is clear, the red block is clear, the hand is empty, the green block is on the table, the orange block is on the table, the red block is on top of the yellow block, and the yellow block is on the table.',
         'the blue block is clear, the red block is clear, the hand is empty, the red block is on top of the orange block, the blue block is on the table, and the orange block is on the table.']

state_holding =[
    'the blue block is clear, the orange block is clear, the hand is holding the red block, the blue block is on the table, and the orange block is on the table.',
    'the cyan block is clear, the red block is clear, the yellow block is in the hand, the hand is holding the yellow block, the cyan block is on top of the orange block, the orange block is on the table, and the red block is on the table.',
    'the yellow block is clear, the orange block is in the hand, the hand is holding the orange block, the yellow block is on top of the blue block, the red block is on top of the yellow block, and the blue block is on the table.',
    'the red block is clear, the orange block is clear, the blue block is in the hand, the yellow block is clear, the hand is holding the blue block, the red block is on the table, the orange block is on the table, and the yellow block is on the table.'
]

pickup_intro = """I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do

Pick up a block
Unstack a block from on top of another block

I have the following restrictions on my actions:
I can only pick up or unstack one block at a time.
I can only pick up or unstack a block if my hand is empty.
I can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.
I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.
I can only unstack a block from on top of another block if the block I am unstacking is clear.
Once I pick up or unstack a block, I am holding the block.

"""

putdown_intro = """I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do

Put down a block
Stack a block on top of another block

I have the following restrictions on my actions:
I can only put down a block that I am holding.
I can only stack a block on top of another block if I am holding the block being stacked.
I can only stack a block on top of another block if the block onto which I am stacking the block is clear.
Once I put down or stack a block, my hand becomes empty.

"""


def get_next_actions_empty(prompt):
    # print("MY PROMPTS for EMPTY!")
    next_action_empty = pickup_intro + \
        "\n".join([
            "[STATEMENT]\nAs initial conditions I have that, " + \
            s + \
            "\n\n[ACTIONS]\n" + \
            "\n".join(make_actions(s))+"\n[ACTIONS END]\n"
            for s in state_empty
        ])
    next_action_empty += "\n[STATEMENT]\nAs initial conditions I have that, <init_state>\n\n[ACTIONS]\n"

    return next_action_empty

def get_next_actions_holding(prompt):
    # print("MY PROMPTS for HOLDING!")
    next_action_holding = putdown_intro + \
        "\n".join([
            "[STATEMENT]\nAs initial conditions I have that, " + \
            s + \
            "\n\n[ACTIONS]\n" + \
            "\n".join(make_actions(s))+"\n[ACTIONS END]\n"
            for s in state_holding
        ])
    next_action_holding += "\n[STATEMENT]\nAs initial conditions I have that, <init_state>\n\n[ACTIONS]\n"

    return next_action_holding
