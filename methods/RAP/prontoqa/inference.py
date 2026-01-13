import itertools
import os
import json
import fire

from dataset import ProntoQADataset
from reasoners import Reasoner

from search_config import ProntoQAConfig
from world_model import ProntoQAWorldModel, ProntoQAAction
from reasoners.algorithm import MCTS
from reasoners.benchmark import ProntoQAEvaluatorFinal
from datetime import datetime


def rap_answer_extractor(mcts_result):
    if mcts_result.trace is None:
        return ""
    else:
        return "\n".join(
            [
                mcts_result.trace[0][i].body
                for i in range(1, len(mcts_result.trace[0]) - 1)
            ]
        )


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def main(
    base_model: str = "hf",
    model_dir: str = "/path/to/model",
    llama_size: str = "7B",
    batch_size: int = 1,
    mem_map: str = "[16, 22]",
    temperature: float = 0.8,
    n_candidates: int = 4,
    quantized: str = "int8",
    **search_algo_params,
):
    import numpy as np
    from reasoners.lm import HFModel

    if base_model == "hf":
        language_model = HFModel(
            model_pth=model_dir,
            tokenizer_pth=model_dir,
            quantized=quantized,
            max_batch_size=batch_size,
        )
        model_name = model_dir.split("/")[-1]
    else:
        raise ValueError(f"base_lm {base_model} is not supported")

    log_dir = (
        f"logs/prontoqa//RAP/{datetime.now().strftime('%m%d%Y-%H%M%S')}_{model_name}"
    )

    with open("methods/CoT/prontoqa/data/example_next_steps.json") as f:
        init_prompt = json.load(f)

    world_model = ProntoQAWorldModel(base_model=language_model)
    search_config = ProntoQAConfig(
        base_model=language_model, temperature=temperature, n_candidates=n_candidates
    )
    search_algo = MCTS(
        output_trace_in_each_iter=True, cum_reward=np.mean, **search_algo_params
    )
    reasoner = Reasoner(
        world_model=world_model, search_config=search_config, search_algo=search_algo
    )

    evaluator = ProntoQAEvaluatorFinal(
        init_prompt=init_prompt["next_steps"],
        sample_prompt_type="rap",
        disable_log=False,
        output_extractor=rap_answer_extractor,
        answer_extractor=lambda x: "\n".join(x.test_example.chain_of_thought[2::2]),
        disable_tqdm=False,
        dataset=ProntoQADataset.from_file("data/prontoqa/345hop_random_true.json"),
    )

    accuracy = evaluator.evaluate(reasoner, num_shot=4, log_dir=log_dir)
    print(f"accuracy: {accuracy}")


if __name__ == "__main__":
    fire.Fire(main)
