import os
import openai
import numpy as np
from typing import Optional, Union, Literal
import time

from .. import LanguageModel, GenerateOutput
from openai import OpenAI

additional_prompt_templates = {
    'ANSWER': "Your response need to be ended with \"So the answer is <Answer>\"\n\n",
    'CONTINUE': "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n",
    'NONE': ""
}

PROMPT_TEMPLATE_ANSWER = "Your response need to be ended with \"So the answer is\"\n\n"
# PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format and logic of previous examples. Don't say any other words.\n\n"
PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n"
PROMPT_TEMPLATE_CUSTOM =  "You are required to generate all possible actions at a given state. Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n" #PROMPT_TEMPLATE_CONTINUE # "Please continue to answer the last question, following the format of previous examples. Don't say any other words. Remember you can only unstack if a block is on top of another block. If the block is on table you have to pick up the block. Similarly, if you want to put down the block on table, the command is put down and if you want to put a block on top of another block, the command is Stack.\n\n" #"Please suggest 3 actions in the last question. You can look at the previous examples and learn how to move the blocks. Don't say any other words.\n\n"

class OpenAIModel(LanguageModel):
    def __init__(self, model:str, max_tokens:int = 2048, temperature=0.0, additional_prompt=None):
        self.model = model
        # assert not (self.model == "gpt-4-1106-preview" or self.model == "gpt-4o"), "Change to gpt-4o-mini for experiments!"
        self.max_tokens = max_tokens
        self.temperature = temperature
        if self.model == "deepseek-reasoner":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", None), base_url="https://api.deepseek.com")
        else:
            self.client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY", None),
            # organization='',
        )
        self.additional_prompt = additional_prompt
        self.tokens_generated = 0
    
    def generate(self,
                prompt: Optional[Union[str, list[str]]],
                max_tokens: int = None,
                top_p: float = 1.0,
                num_return_sequences: int = 1,
                rate_limit_per_min: Optional[int] = 20,
                stop: Optional[str] = None,
                # eos_token_id: Union[None, str, list[str]] = None,
                logprobs: Optional[int] = None,
                temperature = None,
                additional_prompt=None,
                retry = 64,
                custom_prompt_template: str="",
                system_prompt = None,
                **kwargs) -> GenerateOutput:
        
        temperature = self.temperature if temperature is None else temperature
        if isinstance(prompt, list):
            assert len(prompt) == 1
            prompt = prompt[0]
        if additional_prompt is None and self.additional_prompt is not None:
            additional_prompt = self.additional_prompt
        elif additional_prompt is not None and self.additional_prompt is not None:
            print("Warning: additional_prompt set in constructor is overridden.")
        
        
        #if custom_prompt_template != "":
        #    prompt = custom_prompt_template + prompt
        #else:
        #    prompt = additional_prompt_templates[additional_prompt] + prompt
            
        # print(prompt)
        if additional_prompt == "ANSWER":
            prompt = PROMPT_TEMPLATE_ANSWER + prompt
        elif additional_prompt == "CONTINUE":
            prompt = PROMPT_TEMPLATE_CONTINUE + prompt
        elif additional_prompt == "CUSTOM":
            prompt = PROMPT_TEMPLATE_CUSTOM + prompt
        elif additional_prompt == "NONE":
            prompt = prompt

        if max_tokens is None:
            max_tokens = self.max_tokens
        
        if logprobs is None:
            logprobs = 0

        # print("FINAL PROMPT IS ", prompt,flush=True)

        for i in range(1, retry + 1):
            try:
                # sleep several seconds to avoid rate limit
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)
                ### GPT 3.5 and higher use a different API
                if ('o1' in self.model or 'deepseek' in self.model):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    },
                                ],
                            }
                        ]
                    )
                    return GenerateOutput(
                        text=[choice.message.content for choice in response.choices],
                        log_prob=None
                    )
                elif ('gpt-3.5' in self.model) or ('gpt-4' in self.model):
                    messages = [{"role": "user", "content": prompt}]
                    if system_prompt is not None:
                        messages.append({"role": "system", "content": system_prompt})
                    # print("Actual Temperature: ",temperature,"  stop: ",stop, "  model: ", self.model, " \nmessages: ", messages)
                    # print("temperature is : ",temperature,flush=True)
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop
                    )
                    self.tokens_generated += response.usage.completion_tokens
                    return GenerateOutput(
                        text=[choice.message.content for choice in response.choices],
                        log_prob=None
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop,
                        logprobs=0,
                        **kwargs
                    )
                    return GenerateOutput(
                        text=[choice["text"] for choice in response.choices],
                        log_prob=[choice["logprobs"] for choice in response["choices"]]
                    )
            
            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i)
        
        # after 64 tries, still no luck
        raise RuntimeError("GPTCompletionModel failed to generate output, even after 64 tries")
    
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              **kwargs) -> list[np.ndarray]:
        raise NotImplementedError("GPTCompletionModel does not support get_next_token_logits")

    def get_loglikelihood(self,
                          prompt: Union[str, list[str]],
                          **kwargs) -> list[np.ndarray]:
        raise NotImplementedError("GPTCompletionModel does not support get_log_prob")


if __name__ == '__main__':
    model = OpenAIModel(model='gpt-3.5-turbo')
    print(model.generate(['Hello, how are you?', 'How to go to Shanghai from Beijing?']))
