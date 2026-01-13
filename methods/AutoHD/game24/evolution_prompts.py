init_prompt = """\
I need help designing a new heuristic function to solve Game 24 problem. In Game 24, you are given a list of numbers that need to be used with any operation '+', '-', '*', '/' to obtain the goal, which is [24]. You need to design an appropriate heuristic that can be used to solve how far current state is from final state. 

Here is an example problem:
Current State: [4, 4, 6, 8] # (4 + 8) * (6 - 4) = 24
Final State: [24]

Here is another example problem:
Current State: [4, 6] # 4 * 6 = 24
Final State: [24]

Here is another example problem:
Current State: [8, 4, 1, 8] # (8 / 4 + 1) * 8 = 24
Final State: [24]

Here is another example problem:
Current State: [5, 5, 5, 9] # 5 + 5 + 5 + 9 = 24
Final State: [24]

Here is another example problem:
Current State: [24] 
Final State: [24]

Task:
Please design a new heuristic. 
Firstly, describe your heuristic and main steps in one sentence as a python comment. Start the comment with 'Heuristic Description:'

Next, implement it in Python as a function named 'calc_heuristic'. This function should accept 1 argument as show below, and not modify the input:
1. 'Numbers' - The current state, a list of numbers that has to be used to obtain the goal. 

This function should return a single output, heuristic_val, representing the feasibility of reaching the goal, considering any operation using (+, -, *, /) with the current state. For eg: (a+b, a-b, b-a, a*b, a/b, b/a).
It must return 0 if the goal is achieved, i.e. when 24 is the only number left.

Do not give additional explanations. Do not use any tools. Return your response as python code.
"""

evolution_prompt = """\
I need help designing a new heuristic function to solve Game 24 problem. In Game 24, you are given a list of numbers that need to be used with any operation '+', '-', '*', '/' to obtain the goal, which is [24]. You need to design an appropriate heuristic that can be used to solve how far current state is from final state. 

Here is an example problem:
Current State: [4, 4, 6, 8] # (4 + 8) * (6 - 4) = 24
Final State: [24]

Here is another example problem:
Current State: [4, 6] # 4 * 6 = 24
Final State: [24]

Here is another example problem:
Current State: [8, 4, 1, 8] # (8 / 4 + 1) * 8 = 24
Final State: [24]

Here is another example problem:
Current State: [5, 5, 5, 9] # 5 + 5 + 5 + 9 = 24
Final State: [24]

Here is another example problem:
Current State: [24] 
Final State: [24]

<exisiting_heuristics>

Task:
<evolution_type>
Firstly, identify the common idea in the provided heuristics in one sentence as python comment. Start the python comment with 'Common Idea:'
Secondly, based on the backbone idea describe your new heuristic in one sentence as a python comment. Start the python comment with 'Heuristic Description:'
Thirdly, implement it in Python as a function named 'calc_heuristic'. This function should accept 1 argument as show below:
1. 'Numbers' - The current state, a list of numbers that has to be used to obtain the goal. 

This function should return a single output, heuristic_val, representing the feasibility of reaching the goal, considering any operation using (+, -, *, /) with the current state. For eg: (a+b, a-b, b-a, a*b, a/b, b/a).
It must return 0 if the goal is achieved, i.e. when 24 is the only number left.

Do not give additional explanations. Do not use any tools. Return your response as python code.
"""
# Evoltion tyoe to be performed.
evolution_type = {'e1':"Please help me create a new heuristic that has a totally different form from the given ones. \n",
                  'e2':"Please help me create a new heuristic that has a totally different form from the given ones but can be motivated from them. \n",
                  'm1':"Please assist me in creating a new heuristic that has a different form but can be a modified version of the heuristic provided. \n",
                  'm2':"Please identify the main heuristic parameters and assist me in creating a new heuristic that has a different parameter settings of the score function provided. \n",
                  }
types=['e1','e2','m1','m2']