from typing import Generic, TypeVar, Union, NamedTuple, Protocol, Optional, runtime_checkable, Tuple
from typing import Generic, TypeVar, Union, NamedTuple, Protocol, Optional, runtime_checkable
from abc import ABC, abstractmethod

import numpy as np
from transformers import StoppingCriteriaList
from datetime import datetime
import os, sys, pickle
from tqdm import tqdm
import torch
import shutil
from datetime import datetime
import os, sys, pickle
from tqdm import tqdm
import torch
State = TypeVar("State")
Action = TypeVar("Action")
Example = TypeVar("Example")
Trace = tuple[list[State], list[Action]]

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class GenerateOutput(NamedTuple):
    text: list[str]
    log_prob: Optional[list[np.ndarray]] = None


class LanguageModel(ABC):
    @abstractmethod
    def generate(self,
                 inputs: list[str],
                 max_length: Optional[int] = None,
                 max_new_tokens: Optional[int] = None,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 num_return_sequences: int = 1,
                 eos_token_id: Union[None, str, int, list[str, int]] = None,
                 hide_input: bool = True,
                 output_log_probs: bool = False,
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 **kwargs) -> GenerateOutput:
        """Generate text from a list of prompts.

        :param inputs: List of prompts.
        :param max_length: Maximum length of the total output (input + generated).
        :param max_new_tokens: Maximum length of generated tokens. Override max_length.
        :param do_sample: If False, do greedy decoding.
        :param temperature: Temperature for sampling.
        :param top_k: Top-k for sampling.
        :param top_p: Top-p for sampling.
        :param num_return_sequences:
        :param eos_token_id: Token id for end of sentence. Passed *str* will be translated into token_id.
                             Passed *list* will be treated as multiple possible tokens ending the generation.
        :param hide_input: If set true, decode only the generated part.
        :param output_log_probs: If set true, also output the log_probs of each generated token
        :param stopping_criteria:
        """
        ...

    @abstractmethod
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              postprocess: Optional[str] = None,
                              **kwargs) -> list[np.ndarray]:
        """ TODO: doc

        :param prompt:
        :param candidates:
        :param postprocess: optional, can be 'log_softmax' or 'softmax'. Apply the corresponding function to logits before returning
        :return:
        """
        ...

    @abstractmethod
    def get_loglikelihood(self,
                          prefix: str,
                          contents: list[str],
                          **kwargs) -> np.ndarray:
        """Get the log likelihood of the contents given the prefix.

        :param prefix: The prefix to be excluded from the log likelihood.
        :param contents: The contents to evaluate (must include the prefix).
        """
        ...


class WorldModel(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.example = None
        self.prompt = None

    @abstractmethod
    def init_state(self) -> State: ...

    @abstractmethod
    def step(self, state: State, action: Action) -> Union[State, Tuple[State, dict]]:
        """ Returns the next state and optionally an auxiliary data dict

        :param state: The current state
        :param action: The action to take
        :return: The next state and optionally an auxiliary data dict
        """
        ...

    @abstractmethod
    def is_terminal(self, state: State) -> bool: ...

    def update_example(self, example: Example, prompt = None) -> None:        
        if prompt is not None:
            self.prompt = prompt
        self.example = example

class DefaultWorldModel(WorldModel):
    # A default implementation of WorldModel that only 
    # saves the action sequence as the state

    def __init__(self, base_model) -> None:
        super().__init__()
        self.base_model = base_model

    def init_state(self):
        return []

    def step(self, state, action):
        return state + [action], {}

    def is_terminal(self, state):
        # By default the state is never terminal
        return False


class SearchConfig(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.example = None
        self.prompt = None

    @abstractmethod
    def get_actions(self, state: State) -> list[Action]: ...

    def fast_reward(self, state: State, action: Action) -> tuple[float, dict]:
        return 0, {}

    @abstractmethod
    def reward(self, state, action, **kwargs) -> tuple[float, dict]: ...

    def update_example(self, example: Example, prompt = None) -> None:
        if prompt is not None:
            self.prompt = prompt
        self.example = example


@runtime_checkable
class AlgorithmOutput(Protocol[State]):
    terminal_state: State
    trace: Trace


class SearchAlgorithm(ABC):
    def __init__(self, **kwargs): ...

    @abstractmethod
    def __call__(self, world_model: WorldModel, search_config: SearchConfig, **kwargs) -> AlgorithmOutput: ...


class Reasoner(ABC, Generic[State, Action, Example]):
    def __init__(self,
                 world_model: WorldModel[State, Action, Example],
                 search_config: SearchConfig[State, Action, Example],
                 search_algo: SearchAlgorithm) -> None:
        self.world_model = world_model
        self.search_config = search_config
        self.search_algo = search_algo

    def __call__(self, example: Example, prompt = None, **kwargs) -> AlgorithmOutput[State]:
        self.world_model.update_example(example, prompt=prompt)
        self.search_config.update_example(example, prompt=prompt)
        return self.search_algo(self.world_model, self.search_config, **kwargs)

class Evaluator():
    def __init__(self) -> None:
        self.test_at_n = 1
        pass

    @abstractmethod
    def sample_prompt(self,
                      shuffle_prompt,
                      num_shot,
                      sample_prompt_type):
        pass
    
    def evaluate(self,
                 reasoner,
                 shuffle_prompt=True,
                 num_shot=4,
                 resume=0,
                 log_dir=None,
                 tripPlan=False):

        self.dataset = list(self.full_dataset)[resume:]
        try:
            algo_name = reasoner.search_algo.__class__.__name__
        except:
            algo_name = "unknown"

        
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if log_dir is None:
                log_dir = f'logs/{self._dataset_name}_'\
                        f'{algo_name}/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            
            os.makedirs(log_dir, exist_ok=resume > 0)
            os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        
            with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
                print(sys.argv, file=f)

        correct_count = 0

        disable_tqdm = self.disable_tqdm or \
            (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0)
        for i, example in enumerate(tqdm(self.dataset,
                                            initial=resume,
                                            desc=self._dataset_name,
                                            disable=self.disable_tqdm)):
            print()
            print( "------------------------------------------------------------------ ------------------------------------------------------------------ ------------------------------------------------------------------ ", flush = True)
            # print(" EXAMPLE: ", example, flush = True)
            print()
            # print()
            print("INPUT PROCESSOR for EXAMPLE: ", self.input_processor(example), flush = True)
            
            if tripPlan:
                reasoner.world_model.setDays(example["initial_config"]["total_days"])
            algo_output = reasoner(self.input_processor(example),
                                    prompt=self.sample_prompt(
                                        shuffle_prompt=shuffle_prompt,
                                        num_shot=num_shot))
            print("ALGO OUTPUT", algo_output, flush=True)
            
            output = self.output_extractor(algo_output)
            # print()
            # print()
            # print("OUTPUT:\n", output, "\n\n")
            answer = self.answer_extractor(example)
            # print()
            # print()
            # print(" EXAMPLE", example)
    

            # print()
            # print()
            # for key in answer.keys():
            #     print(key , " : ",answer[key])
            # print(" ANSWER", answer)
            # print("------------------------------------------------------------------", flush = True)
            correct = self.eval_output(answer, output)
            correct_count += correct
            accuracy = correct_count / (i + 1)
            log_str = f'Case #{resume + i + 1}: {correct=}, {output=}, {answer=};'\
                        f'{accuracy=:.3f} ({correct_count}/{i + 1})'
            if self.test_at_n and self.test_at_n > 1:
                log_str += f' (test at {self.test_at_n})'
            tqdm.write(log_str)

            if (not self.disable_log) and \
                (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                    print(log_str, file=f)
            
                with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb')  as f:
                    pickle.dump(algo_output, f)
        
        return accuracy

    def batched_evaluate(self, reasoner, shuffle_prompt=True, num_shot=4, resume=0, log_dir=None, batch_size=16):
        self.dataset = list(self.full_dataset)[resume:]
        try:
            algo_name = reasoner.search_algo.__class__.__name__
        except Exception:
            algo_name = "unknown"
            
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if log_dir is None:
                log_dir = f'logs/{self._dataset_name}_{algo_name}/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.makedirs(log_dir, exist_ok=(resume > 0))
            os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
            with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
                print(sys.argv, file=f)

        correct_count = 0
        total_examples = len(self.dataset)
        disable_tqdm = self.disable_tqdm or (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0)
        pbar = tqdm(range(0, total_examples, batch_size),
                    initial=resume,
                    desc=self._dataset_name,
                    disable=disable_tqdm)
        
        idx = resume
        for batch_start in pbar:
            batch_examples = self.dataset[batch_start: batch_start + batch_size]
            batch_inputs = []
            for example in batch_examples:
                batch_inputs.append(self.input_processor(example))
        
                
            algo_outputs = reasoner(batch_inputs, prompt=self.sample_prompt(
                                        shuffle_prompt=shuffle_prompt,
                                        num_shot=num_shot))
            
            for output_idx, example in enumerate(batch_examples):
                print("\n" + "------------------------------------------------------------------", flush=True)
                # print("INPUT PROCESSOR for EXAMPLE: ", self.input_processor(example), flush=True)
                algo_output = algo_outputs[output_idx]
                # print("ALGO OUTPUT", algo_output, flush=True)
                output = self.output_extractor(algo_output)
                answer = self.answer_extractor(example)

                correct = self.eval_output(answer, output)
                correct_count += correct
                idx += 1
                accuracy = correct_count / idx

                log_str = f'Case #{idx}: correct={correct}, output={output}, answer={answer}; accuracy={accuracy:.3f} ({correct_count}/{idx})'
                if self.test_at_n and self.test_at_n > 1:
                    log_str += f' (test at {self.test_at_n})'
                tqdm.write(log_str)

                if (not self.disable_log) and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                    with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                        print(log_str, file=f)
                    with open(os.path.join(log_dir, 'algo_output', f'{idx}.pkl'), 'wb') as f:
                        pickle.dump(algo_output, f)

        return accuracy

    def evaluate_sc(self,
                 reasoner,
                 shuffle_prompt=True,
                 num_shot=4,
                 resume=0,
                 n_sc = 10,
                 log_dir=None):

        self.dataset = list(self.full_dataset)[resume:]
        try:
            algo_name = reasoner.search_algo.__class__.__name__
        except:
            algo_name = "unknown"

        
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if log_dir is None:
                log_dir = f'logs/{self._dataset_name}_'\
                        f'{algo_name}/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
            os.makedirs(log_dir, exist_ok=resume > 0)
            os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        
            with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
                print(sys.argv, file=f)

        correct_count = 0

        disable_tqdm = self.disable_tqdm or \
            (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0)
        for i, example in enumerate(tqdm(self.dataset,
                                            total=resume + len(self.dataset),
                                            initial=resume,
                                            desc=self._dataset_name,
                                            disable=self.disable_tqdm)):
            
            prompt = self.sample_prompt(
                            shuffle_prompt=shuffle_prompt,
                            num_shot=num_shot)
            output_list = []
            save_list = []
            for j in range(n_sc):
                algo_output = reasoner(self.input_processor(example),
                                    prompt=prompt)
                terminal_state = algo_output.terminal_state
                path = ""
                for k in range(len(terminal_state)):
                    path += terminal_state[k].sub_question + " " + terminal_state[k].sub_answer + " "
                save_list.append(path)
                output = self.output_extractor(algo_output)
                output_list.append(output)
                answer = self.answer_extractor(example)
            from collections import Counter
            output = Counter(output_list).most_common(1)[0][0]
            correct = self.eval_output(answer, output)
            correct_count += correct
            accuracy = correct_count / (i + 1)
            log_str = f'Case #{resume + i + 1}: {correct=}, {output=}, {answer=};'\
                        f'{accuracy=:.3f} ({correct_count}/{i + 1})'
            tqdm.write(log_str)

            if (not self.disable_log) and \
                (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                    print(log_str, file=f)
                with open(os.path.join(log_dir, 'algo_output.txt'),'a') as f1:
                    print(save_list, file=f1)
        
        return accuracy
    @abstractmethod
    def eval_output(self, answer, output):
        pass

    