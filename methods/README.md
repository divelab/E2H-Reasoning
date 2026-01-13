Generally, all methods are runnable with any choices of the base models (in `helpers/lm`), e.g., `hugginface`, `openai`, `llama_api`, etc. Simply change the code of model loading `base_model = ...` and it should work.

Note that, when you switch to a new LLM, you may need to manually set the `eos_token_id` or the `stop` for it, so that the generation would stop when the LLM finishes answering the question. Failing to set the `eos_token_id` or `stop` properly may result in slow generation and wrong answer parsing.


