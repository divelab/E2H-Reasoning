from typing import Union, Optional, Callable
import warnings
import copy
# cd85c6b6ad8aac53a9e9b3115fe101a4a4ca4dc66dfe7a28a5610b2da7464e21
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM
import torch
from peft import PeftModel
import numpy as np
# from optimum.bettertransformer import BetterTransformer
from accelerate import infer_auto_device_map, dispatch_model
import time
from .. import LanguageModel,GenerateOutput
from openai import OpenAI
# from fireworks.client import Fireworks   
# from together import Together 
from openai import OpenAI as DeepInfra
import os

additional_prompt_templates = {
    'ANSWER': "Your response need to be ended with \"So the answer is\"\n\n",
    'CONTINUE': "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n",
    'NONE': ""
}

PROMPT_TEMPLATE_ANSWER = "Your response need to be ended with \"The answer is\"\n\n"
# PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format and logic of previous examples. Don't say any other words.\n\n"
PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n"
PROMPT_TEMPLATE_CUSTOM =  "You are required to generate all possible actions at a given state. Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n"

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

class HFModel(LanguageModel):
    def __init__(self, model_pth, tokenizer_pth, device='cuda:0', max_batch_size=1, max_new_tokens=None, max_length=2048, quantized=None, peft_pth=None, load_awq_pth=None,device_map=None, attn_implementation='flash_attention_2', torch_dtype="bfloat16",**kwargs):
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
        self.tokens_generated = 0
        if quantized is not None and "train" in quantized:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pth, add_bos_token=False, lagacy=False, padding_side='left')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pth, lagacy=False, trust_remote_code=True)

        if quantized == "nf4_train":
            print("nf4 quantizing FOR TRAINING.............................")
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                bnb_4bit_use_double_quant=True,
            )
            assert device_map == None
            model = AutoModelForCausalLM.from_pretrained(args.pretrained_model,
                                                trust_remote_code=True,
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config)

            self.model = prepare_model_for_kbit_training(model)
            self.model.config.use_cache = False
        elif quantized == "bf16_train":
            model = AutoModelForCausalLM.from_pretrained(args.pretrained_model,
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True)

        elif quantized == "int8":
            print("int8 quantizing.............................")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_pth,
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map="auto" if device_map is None else device_map,                                    
            )
        elif quantized == "nf4" or quantized  == "fp4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=quantized,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            print("quantizing.............................")

            self.model = AutoModelForCausalLM.from_pretrained(
                model_pth,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        
        elif quantized == "awq":
            try:
                from awq.quantize.pre_quant import apply_awq
                from awq.quantize.quantizer import real_quantize_model_weight
            except ImportError as e:
                print(f'\033[31mError\033[0m: You need to install package awq to use {quantized=}. '
                      'It can be installed with \033[1mpip install -e .[awq]\033[0m under cloned reaonsers repo. '
                      'Refer to https://github.com/mit-han-lab/llm-awq for more details.')
                raise e
            config = AutoConfig.from_pretrained(model_pth, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_pth, trust_remote_code=True, lagacy=False)
            kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
            self.model = AutoModelForCausalLM.from_pretrained(model_pth, config=config, trust_remote_code=True,**kwargs)
            self.model.eval()
            awq_results = torch.load(load_awq_pth, map_location="cpu")
            apply_awq(self.model, awq_results)
            q_config = {
                "zero_point": True,  # by default True
                "q_group_size": 128,  # whether to use group quantization
            }
            real_quantize_model_weight(self.model, w_bit=4, q_config=q_config)
            kwargs = {"max_memory": None}
            device_map = infer_auto_device_map(self.model,no_split_module_classes=["OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "MPTBlock", "DecoderLayer"], **kwargs)
            self.model = dispatch_model(self.model, device_map=device_map)

        else:
            print('Default Loading.............................')
            self.model = AutoModelForCausalLM.from_pretrained(
                model_pth,
                attn_implementation=attn_implementation,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        if peft_pth is not None:
            self.model = PeftModel.from_pretrained(
                self.model, 
                peft_pth,
                torch_dtype=torch.float16
            )                                                                 
        
        self.max_new_tokens = max_new_tokens
        self.max_batch_size = max_batch_size
        self.max_length = max_length
        self.device = device
        # self.model = BetterTransformer.transform(self.model) #not updated yet
        self.model.eval()
        # for old llama tokenizer's config, below is necessary
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        # if torch.__version__ >= "2" and sys.platform != "win32":#need to figure out this line
        #     self.model = torch.compile(self.model) ###make the faketensor bug, an on-going issue in pytorch
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
            additional_prompt: str = "NONE",
            use_api: bool = False,
            stop: Optional[str] = None,
            verbose: bool = True,
            skip_special_tokens: bool = True,
            **kwargs,
        ) -> GenerateOutput:


        if additional_prompt == "ANSWER":
            inputs[0] = PROMPT_TEMPLATE_ANSWER + inputs[0]
        elif additional_prompt == "CONTINUE":
            inputs[0] = PROMPT_TEMPLATE_CONTINUE + inputs[0]
        elif additional_prompt == "CUSTOM":
            inputs[0] = PROMPT_TEMPLATE_CUSTOM + inputs[0]
        elif additional_prompt == "NONE":
            inputs = inputs
    
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
        if eos_token_id_input is not None:
            if not isinstance(eos_token_id_input, list):
                eos_token_id_input = [eos_token_id_input]
            for token in eos_token_id_input:
                if isinstance(token, str):
                    tokenized = self.tokenizer.encode(token, add_special_tokens=False)
                    if len(tokenized) != 1:
                        warnings.warn(f'the eos_token {repr(token)} is encoded into {tokenized} with length != 1, '
                                    f'using {tokenized[-1]} as the eos_token_id')
                    token = tokenized[-1]
                if isinstance(token, int):
                    eos_token_id.append(token)
                else:
                    warnings.warn(f'the eos_token {repr(token)} is neither str nor int, which is ignored')
        eos_token_id.append(self.tokenizer.eos_token_id)

        if do_sample:
            generation_config = GenerationConfig(
                max_length=max_length,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=eos_token_id,
                do_sample = do_sample,
                top_k=top_k,
                top_p=top_p,
            )
        else:
            generation_config = GenerationConfig(
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=eos_token_id,
                do_sample = do_sample,
                top_p=top_p,
            )
        if max_new_tokens is not None:
            if do_sample:
                generation_config = GenerationConfig(
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=eos_token_id,
                do_sample = do_sample,
                top_k=top_k,
                top_p=top_p,
            )
            else:
                generation_config = GenerationConfig(
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=eos_token_id,
                do_sample = do_sample,
                top_p=top_p,
            )
        
        if num_return_sequences > 1:
            assert len(inputs) == 1, 'num_return_sequences > 1 is not supported for multiple inputs'
            inputs = inputs * num_return_sequences
        decoded_list = []
        log_prob_list = []
        # print(inputs, len(inputs))
        start_time = time.time()
        if verbose:
            print('Parameters: ',top_k, top_p, temperature)
        if use_api:
            deepInfra = DeepInfra(
                api_key=os.environ['DEEPINFRA_TOKEN'],
                base_url="https://api.deepinfra.com/v1/openai",
            )
            
        
            for content in inputs:
                messages = [{"role": "user", "content": content}]
                for i in range(1, 5 + 1): # retry
                    try:
                        resp = deepInfra.chat.completions.create(
                            model="meta-llama/Meta-Llama-3.1-405B-Instruct",
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
        else: 
        # quit()
        # start_time = time.time()   
            for start in range(0, len(inputs), self.max_batch_size):
                end = min(start + self.max_batch_size, len(inputs))
                encoded_inputs = self.tokenizer(inputs[start:end], return_tensors='pt', padding=True).to(self.device)
                with torch.inference_mode():
                    generation_output = self.model.generate(
                        **encoded_inputs,
                        generation_config=generation_config,
                        output_scores=output_log_probs,
                        return_dict_in_generate=True,
                    )
                
                decoded = self.tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=skip_special_tokens)
                # print()
                # print("OUTPUTS: ", decoded)
                # print()
                if hide_input:
                    for i in range(end-start):
                        decoded[i] = decoded[i][len(inputs[start+i]):]
                log_prob = None
                if output_log_probs:
                    log_prob = generation_output.scores
                    log_prob_list.extend(log_prob)
                decoded_list.extend(decoded)
        if not output_log_probs:
            log_prob_list = None
        
        # print(decoded_list[])
        if verbose:
            print('Time for generation:', time.time() - start_time)
        # quit()
        # print()
        # print("##################################################")
        # print('Log Prob List:',log_prob_list)
        # print("GENERATING  decoded_list", decoded_list)
        # print()

        return GenerateOutput(decoded_list, log_prob_list)

    @torch.no_grad()
    def get_next_token_logits(
        self,
        prompt: Union[str, list[str]],
        candidates: Union[list[str], list[list[str]]]) -> list[np.ndarray]:
        start_time = time.time()
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(candidates[0], str):
            candidates = [candidates] * len(prompt)
        cand_tokens = []
        for candidate in candidates:
            cand_tokens.append([])
            for cand in candidate:
                token = self.tokenizer.encode(cand, add_special_tokens=False)
                if len(token) != 1:
                    warnings.warn(f'candidate {cand} corresponds to {len(token)} instead of 1')
                cand_tokens[-1].append(token[1] if len(token) > 1 else token[0])
        

        bsz = len(prompt)
        assert bsz <= self.max_batch_size, (bsz, self.max_batch_size)

        tokens = self.tokenizer(prompt, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            all_logits = self.model(**tokens, return_dict=True).logits[:,-1,:].squeeze(1)

        logits = []
        for case_logits, cand in zip(all_logits, cand_tokens):
            logits.append(case_logits[cand].cpu().numpy())
        print('Time Taken:', time.time() - start_time)
        return logits
    
    @torch.no_grad()
    def get_loglikelihood(self, prefix: str, contents: list[str], **kwargs) -> np.ndarray:
        bsz = len(contents)
        assert bsz <= self.max_batch_size, (bsz, self.max_batch_size)
        prompts_tokens = self.tokenizer(contents, return_tensors='pt',add_special_tokens=False, padding=True).to(self.device)
        prefix_tokens = self.tokenizer(prefix, return_tensors='pt',add_special_tokens=False, padding=True).input_ids[0].to(self.device)
        
        for prompt_tokens in prompts_tokens.input_ids:
            assert torch.all(prompt_tokens[: len(prefix_tokens)] == prefix_tokens), (prompt_tokens, prefix_tokens)

        tokens = prompts_tokens
        logits = self.model(**tokens, return_dict=True).logits
        tokens = prompts_tokens.input_ids
        acc_probs = torch.zeros(bsz).to(self.device)
        for i in range(len(prefix_tokens), tokens.shape[1]):
            probs = torch.softmax(logits[:, i-1, :], dim=-1)
            for j in range(bsz):
                if tokens[j, i] != self.tokenizer.pad_token_id:
                    acc_probs[j] += torch.log(probs[j, tokens[j, i]])
        return acc_probs.cpu().numpy()
    
    def get_normalized_loglikelihood(self, prefix: str, contents: list[str], **kwargs) -> np.ndarray:
        bsz = len(contents)
        assert bsz <= self.max_batch_size, (bsz, self.max_batch_size)
        
        # Tokenize the contents and prefix
        prompts_tokens = self.tokenizer(contents, return_tensors='pt', add_special_tokens=False, padding=True).to(self.device)
        prefix_tokens = self.tokenizer(prefix, return_tensors='pt', add_special_tokens=False, padding=True).input_ids[0].to(self.device)
        
        # Ensure the prefix is present in all prompts
        for prompt_tokens in prompts_tokens.input_ids:
            assert torch.all(prompt_tokens[: len(prefix_tokens)] == prefix_tokens), (prompt_tokens, prefix_tokens)
        
        tokens = prompts_tokens
        logits = self.model(**tokens, return_dict=True).logits
        tokens = prompts_tokens.input_ids
        
        acc_probs = torch.zeros(bsz).to(self.device)
        
        # Calculate log likelihood
        for i in range(len(prefix_tokens), tokens.shape[1]):
            probs = torch.softmax(logits[:, i-1, :], dim=-1)
            for j in range(bsz):
                if tokens[j, i] != self.tokenizer.pad_token_id:
                    acc_probs[j] += torch.log(probs[j, tokens[j, i]])
        
        # Normalize by length of the plan
        normalized_acc_probs = acc_probs / (tokens != self.tokenizer.pad_token_id).sum(dim=1).float().to(self.device)
        
        return normalized_acc_probs.detach().cpu().numpy()
    

