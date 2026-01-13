import re


def retrieve_answer_from_dataset(example) -> str:
    return parse_response(example["golden_plan"])


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
