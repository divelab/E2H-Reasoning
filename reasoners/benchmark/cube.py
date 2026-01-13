import datasets
import random
import copy
from reasoners import Evaluator
import reasoners.benchmark.cube_utils as cube_utils


class CubeEvaluator(Evaluator):
    def __init__(
        self,
        output_extractor,
        answer_extractor,
        init_prompt=None,
        disable_log=False,
        disable_tqdm=False,
        sample_prompt_type="l2m",
    ) -> None:
        super().__init__()
        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor

        def debug_input(x):
            sticker_to_state = {
                0: 1,
                1: 2,
                2: 3,
                3: 4,
                4: 5,
                5: 6,
                6: 7,
                7: 8,
                8: 9,
                9: 10,
                10: 11,
                11: 12,
                12: 13,
                13: 14,
                14: 15,
                15: 16,
                16: 17,
                17: 18,
                18: 19,
                19: 20,
                20: 21,
                21: 22,
                22: 23,
                23: 24,
            }

            input_representation = """
            Top:
            {Sticker 0} {Sticker 1}
            {Sticker 2} {Sticker 3}
            
            Right:
            {Sticker 4} {Sticker 5}
            {Sticker 6} {Sticker 7}
            
            Front:
            {Sticker 8} {Sticker 9}
            {Sticker 10} {Sticker 11}
            
            Bottom:
            {Sticker 12} {Sticker 13}
            {Sticker 14} {Sticker 15}

            Left:
            {Sticker 16} {Sticker 17}
            {Sticker 18} {Sticker 19}

            Back:
            {Sticker 20} {Sticker 21}
            {Sticker 22} {Sticker 23}
        """

            for sticker_index, state_number in sticker_to_state.items():
                state_str = f"state_{state_number}"
                state_value = str(x[state_str])
                
                placeholder = f"{{Sticker {sticker_index}}}"
                
                input_representation = input_representation.replace(placeholder, state_value)

            return input_representation

        self.input_processor = debug_input
        self.full_dataset = datasets.load_dataset(
            "csv", data_files="data/cube/data.csv", split="train"
        ).select(range(100))
        self._dataset_name = "rubiks_cube"
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.sample_prompt_type = sample_prompt_type

    def sample_prompt(self, shuffle_prompt=True, num_shot=4):
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
            prompt = {}
            prompt["cot_prefix"] = self.init_prompt["cot_prefix"]
            prompt[sample_prompt_type] = self.init_prompt["cot_prefix"]
        elif sample_prompt_type == "tot":
            prompt = self.init_prompt
            prompt["tot_prefix"] = self.init_prompt["tot_prefix"]
            prompt[sample_prompt_type] = self.init_prompt["tot_prefix"]
        elif sample_prompt_type == "rap":
            ret = copy.deepcopy(self.init_prompt)
            ret["interactive_examples"], ret["useful_examples"] = zip(
                *random.sample(
                    list(zip(ret["interactive_examples"], ret["useful_examples"])),
                    k=num_shot,
                )
            )
            return ret
        elif sample_prompt_type == "o1":
            prompt = {}
            prompt[sample_prompt_type] = self.init_prompt["o1"]
        elif sample_prompt_type == "grace":
            return None

        else:
            raise NotImplementedError
        return prompt

    def eval_output(self, answer, output):
        if output is None:
            return False
        _, old_state = answer
        new_state = cube_utils.doAlgStr(old_state, output)
        print("Cube: ", cube_utils.getCube(new_state))

        return cube_utils.isSolved(new_state)
