dummystate = "the blue block is clear, the orange block is in the hand, the red block is clear, the yellow block is clear, the hand is holding the orange block, the blue block is on the table, the red block is on the table, and the yellow block is on the table."
dummygoal = "the orange block is clear, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the blue block is on the table, the orange block is on the table, and the yellow block is on the table."

init_prompt = """\
I need help designing a new heuristic function to solve blocksworld problem. You are given 2 strings - current state of blocks and a final desired state. The goal is to design an appropriate heuristic that can be used to solve how far current state is from final state. 

Here is an example problem:
Initial state: the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table
Final state: the orange block is clear, the yellow block is clear, the hand is empty, the orange block is on top of the red block, the red block is on top of the blue block, the blue block is on the table, and the yellow block is on the table.



Here is another example problem:
Initial state: the blue block is clear, the orange block is in the hand, the red block is clear, the yellow block is clear, the hand is holding the orange block, the blue block is on the table, the red block is on the table, and the yellow block is on the table.
Final state: the orange block is clear, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the blue block is on the table, the orange block is on the table, and the yellow block is on the table.


Task:
Please design a new heuristic. 
Firstly, describe your heuristic and main steps in one sentence. Start the sentence with 'Heuristic Description:'

Next, implement it in Python as a function named 'calc_heuristic'. This function should accept 2 string inputs as shown above (comma separated with and at the last):
1. 'initial_state' - The current state of blocks .
2. 'final_state' - The final state of blocks.


This function should return one output: 'heuristic_val', which is the heuristic value calculated for the current state of the blocks with respect to final goal state.

Do not give additional explanations. 
"""

evolution_prompt = """\
I need help designing a new heuristic function to solve blocksworld problem. You are given 2 strings - current state of blocks and a final desired state. The goal is to design an appropriate heuristic that can be used to solve how far current state is from final state. 


Here is an example problem:
Initial state: the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the yellow block is on top of the orange block, the blue block is on the table and the orange block is on the table
Final state: the orange block is clear, the yellow block is clear, the hand is empty, the orange block is on top of the red block, the red block is on top of the blue block, the blue block is on the table, and the yellow block is on the table.



Here is another example problem:
Initial state: the blue block is clear, the orange block is in the hand, the red block is clear, the yellow block is clear, the hand is holding the orange block, the blue block is on the table, the red block is on the table, and the yellow block is on the table.
Final state: the orange block is clear, the red block is clear, the yellow block is clear, the hand is empty, the red block is on top of the blue block, the blue block is on the table, the orange block is on the table, and the yellow block is on the table.


<exisiting_heuristics>

Task:
<evolution_type>
Firstly, identify the common idea in the provided heuristics.
Secondly, based on the backbone idea describe your new heuristic in one sentence.
Thirdly, implement it in Python as a function named 'calc_heuristic'. This function should accept 2 string inputs as shown above (comma separated with and at the last):
1. 'initial_state' - The current state of blocks.
2. 'final_state' - The final state of blocks.

This function should return one output: 'heuristic_val', which is the heuristic value calculated for the current state of the blocks.

Do not give additional explanations. 
"""

evolution_type = {'e1':"Please help me create a new heuristic that has a totally different form from the given ones. \n",
                  'e2':"Please help me create a new heuristic that has a totally different form from the given ones but can be motivated from them. \n",
                  'm1':"Please assist me in creating a new heuristic that has a different form but can be a modified version of the heuristic provided. \n",
                  'm2':"Please identify the main heuristic parameters and assist me in creating a new heuristic that has a different parameter settings of the score function provided. \n",
                  }
types=['e1','e2','m1','m2']