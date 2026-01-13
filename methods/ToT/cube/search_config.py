import numpy as np
import scipy
from utils import value_map
from world_model import CubeAction, CubeState

from reasoners import LanguageModel, SearchConfig


class CubeConfig(SearchConfig):
    def __init__(
        self,
        base_model: LanguageModel,
        prompt: dict,
        temperature: float = 0.8,
        n_candidate: int = 5,
        calc_reward: str = "llm",
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.n_candidate = n_candidate
        self.temperature = temperature
        self.calc_reward = calc_reward

    def get_actions(self, state: CubeState) -> list[CubeAction]:
        input_prompt = self.prompt["tot_prefix"]
        input_prompt = input_prompt.replace("{QUESTION}", self.example)
        for i, s in enumerate(state.state_history):
            input_prompt += f"\n{s}\n[Step {i + 2}]:\n[Move]:"

        outputs = self.base_model.generate(
            [input_prompt],
            num_return_sequences=self.n_candidate,
            temperature=self.temperature,
            do_sample=True,
            hide_input=True,
        ).text

        # Filter outputs here
        print("-------------------------------")
        ret = [o.strip() for o in outputs]

        # deduplicate
        ret = list(dict.fromkeys(ret))
        print("ACTIONS Outputs: with temperature  ", self.temperature, "  :  ", ret)
        return ret

    def fast_reward(
        self, state: CubeState, action: CubeAction, useLLMRating: bool = False, **kwargs
    ) -> tuple[float, dict]:
        # LLM rating is required by super class, we are going to use calc_reward
        if self.calc_reward == "llm":
            input_prompt = self.prompt["tot_prefix"]
            input_prompt = input_prompt.replace("{QUESTION}", self.example)
            input_prompt += "".join(["\n" + s for s in state.state_history])

            rating_prompt = self.prompt["rating_prompt"].replace("{input_prompt}", input_prompt).replace("{action}", action)

            rating_response = self.base_model.generate(
                [rating_prompt],
                num_return_sequences=self.n_candidate,
                stop="\n",
                temperature=self.temperature,
                do_sample=True,
                hide_input=True,
            )
            try:
                rating = float(rating_response.text[0].strip())
            except Exception:
                rating = 0
            return rating, {"intuition": rating, "self_eval": rating}

        inputs = self.prompt["tot_prefix"].replace("{QUESTION}", self.example)
        inputs += "".join(["\n" + s for s in state.state_history])

        value_keys = list(value_map.keys())
        logits = self.base_model.get_next_token_logits([inputs], value_keys)[0]
        logits = scipy.special.softmax(logits)
        value = np.sum(logits * np.array(list(value_map.values())))

        return value, {"intuition": value, "self_eval": value}

    def reward(
        self, state: CubeState, action: CubeAction, useLLMRating: bool = False, **kwargs
    ) -> tuple[float, dict]:
        intuition, self_eval = kwargs["intuition"], kwargs["self_eval"]
        return self_eval, {"intuition": intuition, "self_eval": self_eval}

    def update_example(self, example, prompt=None) -> None:
        super().update_example(example, prompt=prompt)
