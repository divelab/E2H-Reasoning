import re
from collections import Counter


def retrieve_answer_from_dataset(example):
    """Retrieve the answer from the dataset.

    Args:
      example: A single example from the dataset.

    Returns:
      A tuple of (day, start_hour, end_hour).
    """
    # print("Retrieving answer from dataset:", example["golden_plan"])
    return parse_response(example["golden_plan"])


def hour_to_num(hr_str):
    return float(hr_str.split(":")[0]) + (0.5 if hr_str.split(":")[1] == "30" else 0.0)


def parse_response(response: str):
    """Parse the response.

    Returns a parsed suggested meeting time in (day, start_hour, end_hour).

    Args:
      response: Raw response from the model.

    Returns:
      A tuple of (day, start_hour, end_hour).
    """
    # print("Parsing response:", response)
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
    # print("Parsed response:", day, start_hour, end_hour)
    return day, hour_to_num(start_hour), hour_to_num(end_hour)


def cot_sc_extractor(algo_output, sc=True):
    # aggregate the results from multiple reasoning chains with majority vote
    # aggregate the results from multiple reasoning chains with majority vote

    if not algo_output:
        return "", -1, -1
    answers = [parse_response(x) for x in algo_output]

    count = Counter(answers)

    most_common = count.most_common(1)
    if most_common:  # Ensure it's not a tie with count > 1
        day, start, end = most_common[0][0]
        return day, start, end

    # If all entries are None, return "", -1, -1
    return "", -1, -1
