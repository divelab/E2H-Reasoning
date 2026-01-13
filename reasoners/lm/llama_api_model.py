from typing import Union, Optional
import warnings
import copy
# cd85c6b6ad8aac53a9e9b3115fe101a4a4ca4dc66dfe7a28a5610b2da7464e21
import torch
import numpy as np
import time
from .. import LanguageModel,GenerateOutput
from openai import OpenAI as DeepInfra
import os

additional_prompt_templates = {
    'ANSWER': "Your response need to be ended with \"So the answer is\"\n\n. Don't give any other words, code or tools. I just want the answer.",
    'CONTINUE': "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n",
    'NONE': ""
}

PROMPT_TEMPLATE_ANSWER = "Your response need to be answered with \"So the answer is\"\n\n. STRICTLY: Don't output any other words or code. \n"
# PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format and logic of previous examples. Don't say any other words.\n\n"
PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format of previous examples. Don't give any other words, code or tools..\n\n"
# PROMPT_TEMPLATE_CUSTOM =  "You are required to generate all possible actions at a given state. Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n"
PROMPT_TEMPLATE_CUSTOM = "Summarize the answer using \"[Restoration Moves]\"\n\n. STRICTLY: Don't output any other words or code. \n"

def prompt_prefix(additional_prompt, prompt):
    if additional_prompt == "ANSWER":
        prompt = PROMPT_TEMPLATE_ANSWER + prompt
    elif additional_prompt == "CONTINUE":
        prompt = PROMPT_TEMPLATE_CONTINUE + prompt
    elif additional_prompt == "CUSTOM":
        prompt = PROMPT_TEMPLATE_CUSTOM + prompt
    elif additional_prompt == "NONE":
        prompt = prompt
    return prompt

class LLaMaApiModel(LanguageModel):
    def __init__(self, 
                 model_pth, 
                 tokenizer_pth, 
                 device='cuda:0', 
                 max_batch_size=1, 
                 max_new_tokens=512, 
                 max_length=2048, 
                 quantized=None, 
                 peft_pth=None, 
                 load_awq_pth=None,
                 device_map=None, 
                 use_api = False,
                 model_id = None,
                 additional_prompt = "NONE",
                 **kwargs):
        super().__init__()
        """
        Initializes a new instance of the `HFModel` class.

        Args:
            model_pth (str): The path to the directory containing the pre-trained model.
            tokenizer_pth (str): The path to the directory containing the pre-trained tokenizer.
            device (str): The device to use for running the model (e.g. "cpu", "cuda").
            max_batch_size (int, optional): The maximum batch size to use for inference. Defaults to 1.
            max_new_tokens (int, optional): The maximum number of new tokens to generate during inference. Defaults to None.
            max_length (int, optional): The maximum length of the input sequence. Defaults to 2048.
            quantized (str, optional): The type of quantization to use for the model. Can be "8bit", "nf4", "fp4", or "awq". Defaults to None.
            peft_pth (str, optional): The path to the directory containing the pre-trained PEFT model. Defaults to None.
            load_awq_pth (str, optional): The path to the directory containing the pre-trained AWQ model. Defaults to None.
        """
        self.max_new_tokens = max_new_tokens
        self.max_batch_size = max_batch_size
        self.max_length = max_length
        self.device = device
        self.model_id = model_id
        self.model = None
        self.use_api = use_api
        self.additional_prompt = additional_prompt
        self.tokens_generated = 0

    def generate(
            self,
            inputs: list[str],
            max_length: Optional[int] = None,
            max_new_tokens: Optional[int] = None,
            do_sample: bool = False,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.9,
            num_return_sequences: int = 1,
            eos_token_id: Union[None, str, int, list[str, int]] = None,
            hide_input: bool = True,
            output_log_probs: bool = False,
            use_together_api: bool = False,
            additional_prompt: str = "NONE", 
            stop = None,
            system_prompt = None,
            **kwargs,
        ) -> GenerateOutput:

        # unify eos_token
        if max_length is None:
            max_length = self.max_length  
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        eos_token_id_input = copy.deepcopy(eos_token_id)
        eos_token_id = []

        if not do_sample or temperature == 0.0:
            warnings.warn('temperature=0.0 is equivalent to greedy search, ')
            do_sample = False
            temperature = 0.01
            top_k = 1 
        if num_return_sequences > 1:
            assert len(inputs) == 1, 'num_return_sequences > 1 is not supported for multiple inputs'
            inputs = inputs * num_return_sequences
        decoded_list = []
        log_prob_list = []
        # print(inputs, len(inputs))
        start_time = time.time()
        print('Parameters: ',top_k, top_p, temperature)
        if additional_prompt == "NONE":
            additional_prompt = self.additional_prompt
        for i in range(len(inputs)):
            inputs[i] = prompt_prefix(additional_prompt, inputs[i])
            # together_client = Together()
        
        deepInfra = DeepInfra(
            api_key=os.environ['DEEPINFRA_TOKEN'],
            base_url="https://api.deepinfra.com/v1/openai",
        )
        
    
        for content in inputs:
            messages = [{"role": "user", "content": content}]
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
                
            for i in range(1, 5 + 1): # retry
                try:
                    resp = deepInfra.chat.completions.create(
                        model=self.model_id,
                        stream=False,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        stop=stop,
                        # top_k=top_k,
                        max_tokens=self.max_new_tokens
                    )
                    self.tokens_generated += resp.usage.completion_tokens
                    decoded_list.append(resp.choices[0].message.content)
                    break
                except Exception as e:
                    print(f"An Error Occured: {e}, sleeping for {i*10} seconds")
                    time.sleep(i*10)
                    deepInfra = DeepInfra(
                        api_key=os.environ['DEEPINFRA_TOKEN'],
                        base_url="https://api.deepinfra.com/v1/openai",
                    ) 
        print('Time for generation:', time.time() - start_time)

        return GenerateOutput(decoded_list, log_prob_list)

    @torch.no_grad()
    def get_next_token_logits(
        self,
        prompt: Union[str, list[str]],
        candidates: Union[list[str], list[list[str]]]) -> list[np.ndarray]:
        raise NotImplementedError("LLaMa API does not support get_next_token_logits")
    
    @torch.no_grad()
    def get_loglikelihood(self, prefix: str, contents: list[str], **kwargs) -> np.ndarray:
        raise NotImplementedError("LLaMa API does not support get_loglikelihood")
    
    def get_normalized_loglikelihood(self, prefix: str, contents: list[str], **kwargs) -> np.ndarray:
        raise NotImplementedError("LLaMa API does not support get_normalized_loglikelihood")
    

