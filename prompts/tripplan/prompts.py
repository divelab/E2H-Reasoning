import json

num_cities = [3, 4, 5, 6, 7, 8, 9, 10]
for num_city in num_cities:
    with open(
        f"methods/CoT/tripplan/data/test_TripPlan-cities-{num_city}.json", "r"
    ) as f:
        data = json.load(f)

    prompt_file = {}
    prompt_file["cot_pool"] = []
    for i, item in enumerate(data):
        print(num_city, i)
        if len(item["cities"].split("**")) < num_city:
            continue
        if len(item["cities"].split("**")) > num_city:
            break
        cot_prompts = [
            f"TASK: {example.strip()}"
            for example in item["prompt_5shot"].split("TASK:")[1:-1]
        ]
        prompt_file["cot_pool"].extend(cot_prompts)
        if i == 0:
            break
    prompt_file["prefix"] = "\n\nTASK: {Question}\nSOLUTION:"
    with open(f"methods/CoT/tripplan/prompts/prompts_pool_{num_city}.json", "w") as f:
        json.dump(prompt_file, f)
