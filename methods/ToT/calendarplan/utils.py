import re

value_map = {"sure": 1, "likely": 0.1, "impossible": 0.0001}


def retrieve_answer_from_dataset(example: str) -> str:
    return parse_response(example)


def hour_to_num(hr_str):
    return float(hr_str.split(":")[0]) + (0.5 if hr_str.split(":")[1] == "30" else 0.0)


def parse_response(response: str):
    time_strs = re.findall(r"[A-Za-z]+, [0-9]+:[0-9]+ - [0-9]+:[0-9]+", response)
    if not time_strs:
        return "", -1, -1
    # If multiple matches are found, return the first one.
    time_str = time_strs[0]
    day, hour_str = (
        time_str.split(",")[0].strip(),
        time_str.split(",")[1].strip(),
    )
    start_hour, end_hour = (
        hour_str.split("-")[0].strip(),
        hour_str.split("-")[1].strip(),
    )
    print("Parsed response:", day, start_hour, end_hour)
    return day, hour_to_num(start_hour), hour_to_num(end_hour)


def tot_extractor(algo_output):
    ans = ""
    try:
        ans = algo_output.terminal_state.action_history[-1]
    except Exception:
        return None

    return parse_response(ans)


def extract_plan_strings(text):
    plan_indexes = [m.start() for m in re.finditer("Plan", text)]
    plans = []
    for i, idx in enumerate(plan_indexes):
        if i == len(plan_indexes) - 1:
            plan = text[idx:]
        else:
            plan = text[idx : plan_indexes[i + 1]]
        plans.append(plan)
    plans_ans = []
    for plan in plans:
        plan = plan.split("\n")
        plans_ans.append(plan[0].strip())
    return plans_ans
