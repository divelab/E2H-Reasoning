import re

def extract_final_answer(sample):
    ans = ''
    if "So the answer is" in sample:
        ## extract answer directly
        # ans_idx = sample.find("So the answer is")
        # ans = re.findall(r'\d+', sample[ans_idx:])
        ans = sample.split('So the answer is')
        if ans:
            ans = ans[-1].strip().split('\n')[0].replace('.', '')
        else:
            ans = ''
    else:
        ## negative word list
        if ' not ' in sample or ' no ' in sample or 'Not ' in sample or 'No ' in sample:
            ans = 'no'
            # print(f"find no: {ans}, {sample}")
        else:
            ans = ''
    return ans

def extract_answer(text):
    match = re.search(r'\b(answer is|answer:) (\byes\b|\bno\b)', text, re.IGNORECASE)
    if match:
        return match.group(2).lower()
    return None

def tot_extractor(algo_output):
    ans = ""
    try:
      sample = algo_output.terminal_state.last_action
    except:
      return None
    
    return extract_answer(sample)

def retrieve_answer_from_dataset(answer):
    return answer

def eval_output(answer, output):
    if output is None:
        return False
    
    # False vs no and True vs yes
    answer = "no" if not answer else "yes"
    
    return answer == output.strip().lower()