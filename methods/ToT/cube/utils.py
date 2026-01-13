import numpy as np

value_map = {"sure": 1, "likely": 0.1, "impossible": 0.0001}


def retrieve_answer_from_dataset(example):
    state_numbers = range(1, 25)
    output = np.array([])
    for state_number in state_numbers:
        state_str = f"state_{state_number}"
        num = example[state_str]
        output = np.append(output, num)
    return example["moves"], output


def tot_extractor(algo_output):
    ans = ""
    try:
        for state in algo_output.terminal_state.state_history[:-1]:
            ans += state + " "
        print("Ans: ", ans)
    except:  # noqa: E722
        return None
    return ans
