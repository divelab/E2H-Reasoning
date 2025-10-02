from datasets import load_dataset, DatasetDict


def get_dataset(
    cfg,
    tokenizer,
    seed
):

    data = load_dataset(cfg.path)
    datasetdict = DatasetDict()

    for split in ["train", "test"]:

        if split not in data:
            raise f"{split} split not in Dataset!!"
        dataset = data[split]

        if cfg.get(f"{split}_size") is not None:
            dataset = dataset.select(range(cfg[f"{split}_size"]))
        if cfg.get(f"{split}_levels") is not None:
            dataset = dataset.filter(lambda example: example["level"] in cfg[f"{split}_levels"])

        dataset = dataset.shuffle(seed=seed)

        dataset = dataset.map(
            prompt_template,
            fn_kwargs={
                    "system_content":cfg.system_content,
                    "user_content":cfg.user_content,
                    "assistant_content":cfg.assistant_content,
                    "tokenizer":tokenizer,
                }
        )

        datasetdict[split] = dataset

    return datasetdict


def prompt_template(
    example,
    system_content,
    user_content,
    assistant_content,
    tokenizer
):
    messages = [
        {
            "role": "system",
            "content": system_content.format(**example)
        },
        {
            "role": "user",
            "content": user_content.format(**example)
        },
        {
            "role": "assistant",
            "content": assistant_content.format(**example)
        },
    ]
    example["prompt"] = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        continue_final_message=True
    )
    return example