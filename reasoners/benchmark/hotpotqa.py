import datasets
import json
from tqdm import tqdm
import torch
import os, pickle
from datetime import datetime
import sys
import random
import copy
from reasoners import Evaluator
from reasoners.benchmark.hotpotutils import normalize_answer, f1_score


class HotpotQAEvaluator(Evaluator):
    def __init__(
        self,
        output_extractor,
        answer_extractor,
        init_prompt=None,
        disable_log=False,
        disable_tqdm=False,
        sample_prompt_type="l2m",
        dataset=None,
        num_shot=None,
    ) -> None:
        super().__init__()
        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.input_processor = lambda x: x["question"]
        self.full_dataset = datasets.load_dataset("json", data_files="data/hotpotQA/data.json", split="train").select(range(100))
        self._dataset_name = "hotpot_qa"
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.sample_prompt_type = sample_prompt_type
        self.num_shot = num_shot

    def sample_prompt(self, shuffle_prompt=True, num_shot=4):
        if self.num_shot is not None:
            num_shot = self.num_shot
        sample_prompt_type = self.sample_prompt_type
        if sample_prompt_type == "l2m":
            prompt = {}
            if shuffle_prompt:
                decomp_examples = random.sample(
                    self.init_prompt["decomposition_pool"], num_shot
                )
                solv_examples = random.sample(
                    self.init_prompt["solving_pool"], num_shot
                )
            else:
                decomp_examples = self.init_prompt["decomposition_pool"][:num_shot]
                solv_examples = self.init_prompt["solving_pool"][:num_shot]
            prompt["decomposition"] = (
                "".join(decomp_examples) + self.init_prompt["composition_prefix"]
            )
            prompt["overall"] = (
                "".join(decomp_examples) + self.init_prompt["overall_prefix"]
            )
            prompt["solving"] = (
                "".join(solv_examples) + self.init_prompt["solving_prefix"]
            )

        elif sample_prompt_type == "cot" or sample_prompt_type == "tasb":
            prompt = self.init_prompt
            key = f"{sample_prompt_type}_pool"
            if shuffle_prompt:
                examples = random.sample(self.init_prompt[key], num_shot)
            else:
                examples = self.init_prompt[key][:num_shot]
            prompt[sample_prompt_type] = "\n".join(examples) + self.init_prompt["prefix"]

        elif sample_prompt_type == "rap":
            ret = copy.deepcopy(self.init_prompt)
            ret["interactive_examples"], ret["useful_examples"] = zip(
                *random.sample(
                    list(zip(ret["interactive_examples"], ret["useful_examples"])),
                    k=num_shot,
                )
            )
            return ret

        elif sample_prompt_type == "grace":
            return None
        elif sample_prompt_type == "o1":
            prompt = {}
            o1_prefix = self.init_prompt["o1_instructions"].replace("{cot_pool_0}", self.init_prompt["cot_pool"][0])
            prompt[sample_prompt_type] = o1_prefix + self.init_prompt["prefix"]

        else:
            raise NotImplementedError
        return prompt

    def eval_output(self, answer, output):
        if output is None:
            return False
        return f1_score(output, answer)[0] > 0.7
