import reasoners.benchmark.bw_utils as utils
from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners import WorldModel, LanguageModel, SearchConfig
from typing import Literal
import reasoners.benchmark.bw_utils as utils
from world_model import BWState, BWAction

class BWConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 temperature: float = 0.8,
                 n_candidate: int = 4,
                 calc_reward: Literal['sampling', 'logits'] = 'sampling') -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.n_candidate = n_candidate
        self.temperature = temperature
        self.calc_reward = calc_reward

    def get_actions(self, state: BWState) -> list[BWAction]:
        prompts = self.prompt["icl"].replace("<action>", "\n".join(state.action_history + [""])) \
            .replace("<init_state>", utils.extract_init_state(self.example)) \
            .replace("<goals>", utils.extract_goals(self.example, return_raw=True))
        # print(prompts)
        outputs = self.base_model.generate([prompts],
                                          num_return_sequences=self.n_candidate,
                                          #max_length=20,
                                          eos_token_id=["\n[", "\n", ],
                                          temperature=self.temperature,
                                          do_sample=True,
                                          hide_input=True).text
        outputs = [output.split("\n")[0].strip() for output in outputs]
        print("Outputs: ", outputs)
        # deduplicate
        outputs = list(dict.fromkeys(outputs))
        print("state.action_history: ", state.action_history)
        print("Outputs: ", outputs)
        return outputs


    def fast_reward(self, state: BWState, action: BWAction) -> tuple[float, dict]:
        if self.calc_reward == "sampling":
            action_history_str = "\n".join(state.action_history + [""])
            rating_prompt = f"{self.prompt['intro']}\nNow given\ninitial state: {utils.extract_init_state(self.example)}\n" \
                f"and the goals:\n{utils.extract_goals(self.example, return_raw=True)}\n" \
                f"and the action history:\n{action_history_str}\n" \
                f"Rate the action:\n'{action}'\non a scale from 1 to 10. Provide only a number in response."
            rating_response = self.base_model.generate([rating_prompt])
            try:
                rating = float(rating_response.text[0].strip())
            except:
                rating = 0
            return rating, {'intuition': rating, 'self_eval': rating}
        elif self.calc_reward == "logits":
            inputs = self.prompt["icl"].replace("<action>", "\n".join(state.action_history + [""])) \
                .replace("<init_state>", utils.extract_init_state(self.example)) \
                .replace("<goals>", utils.extract_goals(self.example, return_raw=True))[:-1]
            intuition = self.base_model.get_loglikelihood(inputs+ "\n", [inputs + "\n" + action])[0]

            self_eval_prompt = self.prompt["self-eval"].replace("<init_state>", 
                                                                utils.extract_init_state(self.example)) \
                                                        .replace("<goals>", utils.extract_goals(self.example, return_raw=True)) \
                                                        .replace("<action>", action)
            self_eval = self.base_model.get_loglikelihood(self_eval_prompt, 
                [self_eval_prompt + "good"])[0]
            print('Fast Reward Logits', self_eval, intuition)
            # quit()
        

        return intuition + self_eval, {'intuition': intuition, "self_eval": self_eval}

    def reward(self, state: BWState, action: BWAction, **kwargs) -> tuple[float, dict]:
        # since these two rewards are fast, we can just return the reward
        intuition, self_eval = kwargs['intuition'], kwargs['self_eval']
        if self.calc_reward == "sampling":
            return self_eval, {'intuition': intuition, "self_eval": self_eval}
        elif self.calc_reward == "logits":
            return intuition + self_eval, {'intuition': intuition, "self_eval": self_eval}

    def update_example(self, example, prompt=None) -> None:
        super().update_example(example, prompt=prompt)