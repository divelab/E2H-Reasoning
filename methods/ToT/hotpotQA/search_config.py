import numpy as np
import scipy
from utils import value_map
from world_model import HotpotAction, HotpotState

from reasoners import LanguageModel, SearchConfig


class HotpotConfig(SearchConfig):
    def __init__(
        self,
        base_model: LanguageModel,
        prompt: dict,
        temperature: float = 0.8,
        n_candidate: int = 3,
        depth_limit: int = 10,
        calc_reward: str = "llm",
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.n_candidate = n_candidate
        self.temperature = temperature
        self.depth_limit = depth_limit
        self.calc_reward = calc_reward

    def get_actions(self, state: HotpotState) -> list[HotpotAction]:
        input_prompt = self.prompt["cot"]
        input_prompt = input_prompt.replace("{QUESTION}", self.example)
        input_prompt += "".join([" " + s for s in state.state_history])
        input_prompt += self.prompt["decision_prompt"]

        print("Prompt: ", input_prompt)

        outputs = []
        if len(state.state_history) == self.depth_limit - 1:
            input_prompt = self.prompt["cot"].replace("{QUESTION}", self.example)
            input_prompt += "".join([". " + s for s in state.state_history])
            input_prompt += self.prompt["final_answer_prompt"]
            print("Prompt: ", input_prompt)
            outputs = self.base_model.generate(
                [input_prompt],
                num_return_sequences=self.n_candidate,
                temperature=self.temperature,
                do_sample=True,
                hide_input=True,
            ).text
        else:
            outputs = self.base_model.generate(
                [input_prompt],
                num_return_sequences=self.n_candidate,
                stop=".",
                temperature=self.temperature,
                do_sample=True,
                hide_input=True,
            ).text

        print("Action history: ", "".join(["  " + s for s in state.state_history]))
        # Filter outputs here
        print("-------------------------------")
        # deduplicate
        ret = list(dict.fromkeys(outputs))
        print("ACTIONS Outputs: with temperature  ", self.temperature, "  :  ", ret)
        return ret

    def fast_reward(
        self, state: HotpotState, action: HotpotAction, useLLMRating: bool = False
    ) -> tuple[float, dict]:
        if self.calc_reward == "llm":
            input_prompt = self.prompt["cot"]
            input_prompt = input_prompt.replace("{QUESTION}", self.example)
            input_prompt += "".join(["\n" + s for s in state.state_history])

            rating_prompt = self.prompt["rating_prompt"].replace("{input_prompt}", input_prompt).replace("{action}", action)
            print("Rating prompt: ", rating_prompt)
            rating_response = self.base_model.generate(
                [rating_prompt],
                num_return_sequences=self.n_candidate,
                stop=".",
                temperature=self.temperature,
                do_sample=True,
                hide_input=True,
            )
            try:
                rating = float(rating_response.text[0].strip())
            except Exception:
                rating = 0
            return rating, {"intuition": rating, "self_eval": rating}

        inputs = self.prompt["cot"].replace("{QUESTION}", self.example)
        inputs += "".join(["\n" + s for s in state.state_history])

        value_keys = list(value_map.keys())
        logits = self.base_model.get_next_token_logits([inputs], value_keys)[0]
        logits = scipy.special.softmax(logits)
        value = np.sum(logits * np.array(list(value_map.values())))

        return value, {"intuition": value, "self_eval": value}

    def reward(
        self,
        state: HotpotState,
        action: HotpotAction,
        useLLMRating: bool = False,
        **kwargs,
    ) -> tuple[float, dict]:
        intuition, self_eval = kwargs["intuition"], kwargs["self_eval"]
        return self_eval, {"intuition": intuition, "self_eval": self_eval}

    def update_example(self, example, prompt=None) -> None:
        super().update_example(example, prompt=prompt)
