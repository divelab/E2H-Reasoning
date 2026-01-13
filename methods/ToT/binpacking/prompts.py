# 5-shot
output_prompt = '''You are tasked with solving the Bin Packing Problem. You are given items of certain sizes and the bin size.
The goal is to find the minimum number of bins required to pack all items such that the item sizes in each bin does not exceed the given bin size.

Input: 
- Item Sizes: [6, 2, 1, 8, 7]
- Bin Size: 10
Steps:
One bin for [6, 2, 1] (left: [8, 7])
One bin for [8] (left: [7])
One bin for [7] (left: [])
Answer: The answer is 3, and the bins are [[6, 2, 1], [8], [7]].

Input: 
- Item Sizes: [8, 4, 4, 6, 7]
- Bin Size: 13
Steps:
One bin for [6, 7] (left: [8, 4, 4])
One bin for [4, 4] (left: [8])
One bin for [8] (left: [])
Answer: The answer is 3, and the bins are [[6, 7], [4, 4], [8]].

Input: 
- Item Sizes: [9, 8, 4, 4, 4, 6]
- Bin Size: 12
Steps:
One bin for [8, 4] (left: [9, 4, 4, 4, 6])
One bin for [4, 4] (left: [9, 6])
One bin for [9] (left: [6])
One bin for [6] (left: [])
Answer: The answer is 4, and the bins are [[8, 4], [4, 4], [9], [6]].

Input:
- Item Sizes: [8, 3, 9, 1, 3, 1, 3]
- Bin Size: 17
Steps:
One bin for [8, 9] (left: [3, 1, 3, 1, 3])
One bin for [3,1,3,1,3] (left: [])
Answer: The answer is 2, and the bins are [[8, 9], [3, 1, 3, 1, 3]].

Input:
- Item Sizes: {item_sizes}
- Bin Size: {bin_size}
Steps:
{history}
Answer:'''

# 1-shot
propose_prompt = '''Input:
- Items: [6, 2, 1, 8, 7]
- Bin Capacity: 10
Possible next steps:
One bin for [6, 2, 1] (left: [8, 7])
One bin for [7, 2, 1] (left: [8, 6])
One bin for [8, 2] (left: [6, 7])
One bin for [6, 2] (left: [1, 8, 7])
One bin for [7, 2] (left: [1, 8, 6])
One bin for [6, 1] (left: [2, 8, 7])
One bin for [8] (left: [6, 7, 2, 1])
One bin for [8, 1] (left: [6, 2, 7])
One bin for [7] (left: [6, 8, 2, 1])

Input: 
- Items: {item_sizes}
- Bin Capacity: {bin_size}
Possible next steps:
'''

value_prompt = '''Evaluate if the given items of size in a bin fit within the given bin capacity (sure/impossible).
Bin - [6, 7] capacity - 10
6+7 = 13 > 10
impossible

Bin - [4, 5] capacity - 11
4+5 = 9 < 11
sure

Bin - [9] capacity - 12
9 < 12
sure

Bin - [6, 1, 1, 2] capacity - 8
6+1+1+2 = 10 > 8
impossible

Bin - [1, 2, 5, 2] capacity - 8
1+2+5+2 = 12 > 8
impossible

Bin - [3, 1, 3, 1, 3] capacity - 17
3+1+3+1+3 = 12 < 17
sure

Bin - [3, 3, 4, 2] capacity - 15
3+3+4+2 = 11 < 15
sure

Bin - [3, 3, 4, 2, 2] capacity - 14
3+3+4+2+2 = 12 < 14
sure

Bin - {bin} capacity - {capacity}
'''

value_last_step_prompt = '''You are tasked with solving the Bin Packing Problem. You are given items of certain sizes and the bin size.
The goal is to find the minimum number of bins required to pack all items such that the item sizes in each bin does not exceed the given bin size. 

Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. if the items in the each bin fit within the bin capacity, and the number of bins is the minimum.

Input: 
- Item Sizes: [6, 2, 1, 8, 7]
- Bin Size: 10
Answer: The answer is 3, and the bins are [[6, 2, 1], [8], [7]].
Judge: sure

Input: 
- Item Sizes: [8, 4, 4, 6, 7]
- Bin Size: 13
Answer: The answer is 2, and the bins are [[6, 7], [4, 4, 8]]
Judge: impossible

Input:
- Item Sizes: [8, 3, 9, 1, 3, 1, 3]
- Bin Size: 17
Answer: The answer is 2, and the bins are [[8, 9], [3, 1, 3, 1, 3]].
Judge: sure

Input: 
- Item Sizes: [9, 8, 4, 4, 4, 6]
- Bin Size: 12
Answer: The answer is 4, and the bins are [[8, 3], [4, 9], [4, 4], [6]].
Judge: impossible

Input:
- Item Sizes: [12, 9, 7, 3, 2, 4]
- Bin Size: 15
Answer: The answer is 3, and the bins are [[12, 2], [9, 7], [4, 3, 2]].
Judge: impossible

Input:
- Item Sizes: [9, 5, 9, 4, 1, 3, 5]
- Bin Size: 12
Answer: The answer is 4, and the bins are [[9, 3], [5, 5, 1], [4], [9]].
Judge: sure

Input: 
- Item Sizes: {item_sizes}
- Bin Size: {bin_size}
Answer: {answer}
Judge:'''

value_name = ['sure', 'impossible']
value_map = {'sure': 1, 'impossible': 0.0001}
