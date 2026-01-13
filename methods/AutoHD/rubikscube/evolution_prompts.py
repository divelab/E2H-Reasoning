init_prompt = """I need help designing a new heuristic function to solve 2x2 Pocket Cube. The problem is defined as the following. Your task is to restore a scrambled 2x2 Rubik's Cube to its original state. All the given problems can be solved in 1 to 4 moves. You cannot exceed more than 11 moves. Provide the sequence of moves required for the restoration. Please follow the instructions and rules below to complete the solving:
1. A 2x2 Pocket Cube has six faces, namely: [Upper, Front, Bottom, Left, Right, Back] Each consisting of a 2x2 grid of squares, with each square having its own color.
2. Colors in the Cube are represented in numbers: [0, 1, 2, 3, 4, 5]
3. The Cube's state is represented as an array of 24 elements. For instance, [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5]. The Cube's state is represented as a 24-element array, where each group of 4 consecutive elements corresponds to a face of the cube in the following order: Upper face: Elements at indices 0 to 3. Right face: Elements at indices 4 to 7. Front face: Elements at indices 8 to 11. Down face: Elements at indices 12 to 15. Left face: Elements at indices 16 to 19. Back face: Elements at indices 20 to 23. Each element within a group represents the color or state of a specific square on that face.
4. A restoration of a Pocket Cube is to move squares in each face to have same numbers. Some example Restored States are [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5]. 
You must make move to the Cube to achieve a Restored State, not limited to the above one. Note that we just need each face to have same numbers, no matter which face has which color.
Task:
Please design a new heuristic. 
Firstly, describe your heuristic and main steps in one sentence. Start the sentence with 'Heuristic Description:'

Next, implement it in Python as a function named 'calc_heuristic'. This function should accept 1 input as shown above :
1. 'State' - The current state of 2x2 Cube, which is a numpy array.

This function should return one output: 'heuristic_val', which is the heuristic value calculated for the current state of the 2x2 Cube with respect to one of restored states.

Do not give additional explanations. """

evolution_prompt = """\
I need help designing a new heuristic function to solve 2x2 Pocket Cube. The problem is defined as the following. Your task is to restore a scrambled 2x2 Rubik's Cube to its original state. All the given problems can be solved in 1 to 4 moves. You cannot exceed more than 11 moves. Provide the sequence of moves required for the restoration. Please follow the instructions and rules below to complete the solving:
1. A 2x2 Pocket Cube has six faces, namely: [Upper, Front, Bottom, Left, Right, Back] Each consisting of a 2x2 grid of squares, with each square having its own color.
2. Colors in the Cube are represented in numbers: [0, 1, 2, 3, 4, 5]
3. The Cube's state is represented as an array of 24 elements. For instance, [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5]. The Cube's state is represented as a 24-element array, where each group of 4 consecutive elements corresponds to a face of the cube in the following order: Upper face: Elements at indices 0 to 3. Right face: Elements at indices 4 to 7. Front face: Elements at indices 8 to 11. Down face: Elements at indices 12 to 15. Left face: Elements at indices 16 to 19. Back face: Elements at indices 20 to 23. Each element within a group represents the color or state of a specific square on that face.
4. A restoration of a Pocket Cube is to move squares in each face to have same numbers. Some example Restored States are [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5]. 
You must make move to the Cube to achieve a Restored State, not limited to the above one. Note that we just need each face to have same numbers, no matter which face has which color.

<exisiting_heuristics>

Task:
<evolution_type>
Firstly, identify the common idea in the provided heuristics.
Secondly, based on the backbone idea describe your new heuristic in one sentence.
Thirdly, implement it in Python as a function named 'calc_heuristic'. This function should accept 1 input as shown above :
1. 'State' - The current state of 2x2 Cube, which is a numpy array.

This function should return one output: 'heuristic_val', which is the heuristic value calculated for the current state of the 2x2 Cube with respect to one of restored states.

Do not give additional explanations. 
"""
# Evoltion tyoe to be performed.
evolution_type = {'e1':"Please help me create a new heuristic that has a totally different form from the given ones. \n",
                  'e2':"Please help me create a new heuristic that has a totally different form from the given ones but can be motivated from them. \n",
                  'm1':"Please assist me in creating a new heuristic that has a different form but can be a modified version of the heuristic provided. \n",
                  'm2':"Please identify the main heuristic parameters and assist me in creating a new heuristic that has a different parameter settings of the score function provided. \n",
                  }
types=['e1','e2','m1','m2']