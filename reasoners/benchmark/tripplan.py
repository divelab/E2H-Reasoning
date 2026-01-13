from typing import Any
from reasoners import Evaluator
import random
import json
import os, sys, pickle
from tqdm import tqdm
import torch
from datetime import datetime
import shutil

# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
def load_dataset(dataset_path):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset


def calc_fitness_score(output, answer):
    fitness_score = 0
    min_length = min(len(output), len(answer))
    for i in range(min_length):
        op_city, op_day = output[i]
        a_city, a_day = answer[i]
        city_distance = 0

        if op_city != a_city:
            city_distance = 10
            days_distance = abs(op_day + a_day)
        else:
            days_distance = abs(op_day - a_day)

        fitness_score += city_distance + days_distance

    for j in range(min_length, len(answer)):
        a_city, a_day = answer[j]
        fitness_score += 10 + a_day + 1

    return fitness_score


class TripPlanEvaluator(Evaluator):
    def __init__(
        self,
        output_extractor,
        answer_extractor,
        init_prompt=None,
        disable_log=False,
        disable_tqdm=False,
        input_processor=lambda x: x["prompt_0shot"],
        dataset_path: str = "",
        sample_prompt_type="l2m",
        heuristic_search=False,
        num_cities = 3,
    ) -> None:
        super().__init__()
        self.parsed_plan = None
        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.full_dataset = load_dataset(dataset_path)[:50]
        self.sample_prompt_type = sample_prompt_type
        self._dataset_name = "tripplan"
        self.input_processor = input_processor
        self.heuristic_search = heuristic_search
        self.num_cities=num_cities

    def sample_prompt(self, shuffle_prompt, num_shot):
        sample_prompt_type = self.sample_prompt_type
        if sample_prompt_type == "cot" or sample_prompt_type == "tot":
            prompt = {}
            pool_key = f"cot_pool_{self.num_cities}"
            if shuffle_prompt:
                examples = random.sample(self.init_prompt[pool_key], num_shot)
            else:
                examples = self.init_prompt[pool_key][:num_shot]
                
            prompt["tot_prefix"] = self.init_prompt["tot_prefix"]
            prompt['self-eval'] = self.init_prompt['self-eval']
            prompt[sample_prompt_type] = "".join(examples) + self.init_prompt["prefix"]
        elif sample_prompt_type == "o1":
            prompt = {}
            prompt[sample_prompt_type] = self.init_prompt["instructions"] + self.init_prompt["o1-prompt"] + self.init_prompt["prefix"]
        elif sample_prompt_type == "beamstar":
            prompt = {}
            prompt_string = self.init_prompt["cot_prefix"]
            if shuffle_prompt:
                examples = random.sample(self.init_prompt["actions_cot_pool"], num_shot)
            else:
                examples = self.init_prompt["actions_cot_pool"][:num_shot]

            prompt[sample_prompt_type] = (
                prompt_string + "".join(examples) + self.init_prompt["prefix"]
            )
        else:
            return NotImplementedError

        return prompt

    def eval_output(self, answer, output: list[Any]):
        if self.heuristic_search:  # Return Fitness Value.
            return calc_fitness_score(output, answer)
        num_stays = min(len(answer), len(output))
        num_match = 0
        for i in range(num_stays):
            if answer[i][0] == output[i][0] and answer[i][1] == output[i][1]:
                num_match += 1
            else:
                break
        hard_score = 0.0 if num_match / len(answer) < 1.0 else 1.0
        return hard_score