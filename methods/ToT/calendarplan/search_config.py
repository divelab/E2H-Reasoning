import numpy as np
import scipy
from utils import extract_plan_strings, value_map
from world_model import CalendarAction, CalendarStateTot

from reasoners import LanguageModel, SearchConfig


class CalendarPlanToTConfig(SearchConfig):
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
        self.example: str = None
        self.prompt = prompt
        self.n_candidate = n_candidate
        self.temperature = temperature
        self.calc_reward = calc_reward

    def get_actions(self, state: CalendarStateTot) -> list[CalendarAction]:
        prompt: str = self.prompt["tot"]
        prompt = prompt.replace("{QUESTION}", self.example)
        if state.step_idx == 0:
            prompt += self.prompt["tot_plan_generation_prompt"]
            llm_output = self.base_model.generate(
                [prompt],
                num_return_sequences=1,
                eos_token_id=None,
                temperature=self.temperature,
                do_sample=True,
                hide_input=True,
            ).text
            llm_output = extract_plan_strings(llm_output[0])
        else:
            prompt += self.prompt["tot_solution_generation_prompt"]
            llm_output = self.base_model.generate(
                [prompt],
                num_return_sequences=self.n_candidate,
                eos_token_id=None,
                temperature=self.temperature,
                do_sample=True,
                hide_input=True,
            ).text
        print("Outputs: ", llm_output)
        outputs = llm_output
        return outputs

    def fast_reward(
        self, state: CalendarStateTot, action: CalendarAction, useLLMRating=True
    ) -> tuple[float, dict]:
        if self.calc_reward == "llm":
            input_prompt = self.prompt["tot"]
            input_prompt = input_prompt.replace("{QUESTION}", self.example)
            input_prompt += "".join(["\n" + s for s in state.action_history])
            if state.step_idx == 0:
                rating_prompt = self.prompt["tot_rating_prompt_plan"].replace("{input_prompt}", input_prompt).replace("{action}", action)
            elif state.step_idx == 1:
                rating_prompt = self.prompt["tot_rating_prompt_solution"].replace("{input_prompt}", input_prompt).replace("{action}", action)
            # print("Rating prompt: ", rating_prompt)

            rating_response = self.base_model.generate(rating_prompt)
            try:
                rating = float(rating_response.text[0].strip())
            except Exception:
                rating = 0
            return rating, {"intuition": rating, "self_eval": rating}

        inputs = self.prompt["tot"].replace("{QUESTION}", self.example)
        inputs += "".join(["\n" + s for s in state.action_history])

        value_keys = list(value_map.keys())
        logits = self.base_model.get_next_token_logits([inputs], value_keys)[0]
        logits = scipy.special.softmax(logits)
        value = np.sum(logits * np.array(list(value_map.values())))

        return value, {"intuition": value, "self_eval": value}

    def reward(self, state, action, **kwargs) -> tuple[float, dict]:
        return kwargs["intuition"] + kwargs["self_eval"], kwargs
