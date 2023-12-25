from typing import Dict
from abc import abstractmethod
from vllm import LLM, SamplingParams
import gc
import torch
import openai
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
#from accelerate import load_checkpoint_and_dispatch

#import text_generation as tg


class ResponseModel:
    @abstractmethod
    def get_response(input_text, **kwargs) -> Dict[str, any]:
        return NotImplementedError
    

class AutoHFResponseModel(ResponseModel):
    def __init__(self, 
                 model_name: str,
                 **kwargs) -> None:
        
        self.device = torch.device("cuda")

        # Load model
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)
        self.model = load_checkpoint_and_dispatch(model, model_name, device_map = "auto", no_split_module_classes=model._no_split_modules, dtype=torch.float16)
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_response(self, input_text: str, **kwargs) -> Dict[str, any]:
        encoded_inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        # Setting generation config; greedy decoding by default
        generation_cfg = {'do_sample': kwargs.get('do_sample', False),
                          'temperature': kwargs.get('temperature', 0),
                          'max_new_tokens': kwargs.get('max_new_tokens', 128),
                          'use_cache': True}

        with torch.no_grad():
            output = self.model.generate(**encoded_inputs, **generation_cfg)
        sequences = output.sequences
        
        all_decoded_text = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        
        completions = []
        for decoded_text in all_decoded_text:
            completions.append({"text": decoded_text})

        return {"completions": completions}
    

class TGIResponseModel(ResponseModel):
    def __init__(self,
                 api_base,
                 timeout: int = 10000,
                 **kwargs) -> None:
        self._client = tg.Client(api_base, timeout=timeout)

    def get_response(self, input_text: str, **kwargs) -> Dict[str, any]:

        # Generation cfg
        generation_cfg = {'best_of': kwargs.get('best_of', 1),
                          'do_sample': kwargs.get('do_sample', False),
                          'temperature': kwargs.get('temperature', None),
                          'max_new_tokens': kwargs.get('max_new_tokens', 128)}
        outputs = self._client.generate(input_text, **generation_cfg)
        generated_text = outputs.generated_text

        return {"completions": [generated_text]}

class Chatgpt:
    def __init__(self):
        openai.api_type = "azure"
        openai.api_version = "2023-03-15-preview" 
        openai.api_base = "https://sslgroupservice.openai.azure.com/"
        openai.api_key = '97f22a7a32ff4ff4902003896f247ca2'

    def inference(self, messages):
        output = openai.ChatCompletion.create(
            engine="gpt-4",
            messages=messages,
            temperature=0.3
            )
        return output['choices'][0]['message']

class OpenAIResponseModel(ResponseModel):
    def __init__(self,
                 **kwargs):
        self.model = Chatgpt()

    def get_response(self, input_text: str, history=False, **kwargs) -> Dict[str, any]:
        messages=[{"role": "system", "content": ""},]
        try:
            messages.append({"role": "user", "content": input_text})
            output = dict(self.model.inference(messages))
            if history:
                messages.append(output)
            else:
                del messages[1]
            
            generated_text = output["content"]

            print(f"------------ out start ------------ \n")
            print(generated_text)
            print(f"------------ out end ------------ \n")
            time.sleep(2)

        except Exception as e: 
            print(e)
            generated_text = ''
            
        return generated_text
"""
class OpenAIResponseModel(ResponseModel):
    def __init__(self, 
                 api_base: str = None, 
                 api_key: str = None,
                 api_type: str = None,
                 api_version: str = None,
                 engine: str = None,
                 **kwargs):
        
        self.api_base = api_base
        self.api_key = api_key
        self.api_type = api_type
        self.api_version = api_version
        self.engine = engine

    def get_response(self, input_text: str, **kwargs) -> Dict[str, any]:
        def _do_it():
            openai.organization = None
            openai.api_key = self.api_key
            openai.api_base = self.api_base
            openai.api_type = self.api_type
            openai.api_version = self.api_version

            raw_request = {'engine': self.engine,
                           'messages': [{"role": "user", "content": input_text}],
                           'temperature': kwargs.get('temperature', None)}
            extra_cfg=dict(headers={'Host': 'mlop-azure-gateway.mediatek.inc', 'X-User-Id': 'mtk53598'})
            raw_request.update(**extra_cfg)
            return openai.ChatCompletion.create(**raw_request)
        
        try:
            outputs = _do_it()
        except openai.error.OpenAIError as e:
            error: str = f"OpenAI error: {e}"
            print(error)
            
        completions = [c['message']['contnet'] for c in outputs['choices']]
        
        return {"completions": completions}
"""


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_token_ids = [0]
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class Fox(ResponseModel):
    def __init__(self,
                version: str = None, 
                 **kwargs) -> None:
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if version == "v0":
            model_dir = '/data1/user_guanru/llm_evaluation/model/foxbrain_v1.1/llama_13b_8k_Mix_Zh_general_domain_0_47_epc_(v0)'
        elif version == "v1":
            model_dir = '/data1/user_guanru/llm_evaluation/model/foxbrain_v1.2/zh_llama2_13b_Foxbrain_v1_1_10_epc_v1'
        elif  version == "v2":
            model_dir = '/data1/user_guanru/llm_evaluation/model/foxbrain_v1.2/Zh_llama2_13b_Foxbrain_v1_1_10_epc_update_v2'
        elif  version == "v3":
            model_dir = '/data1/user_guanru/llm_evaluation/model/foxbrain_v1.2/Zh_llama2_13b_Foxbrain_v1_16_3epc_v3'
        elif  version == "v4":
            model_dir = '/data1/user_guanru/llm_evaluation/model/foxbrain_v1.2/Zh_llama2_13b_Foxbrain_v1_13_5epc_test_1_v4'
        elif version == "v5":
            model_dir = '/data1/user_guanru/llm_evaluation/model/foxbrain_v1.2/Zh_llama2_13b_Foxbrain_v1_1_12_epc_update'
        elif version == "v6":
            model_dir = '/data1/user_guanru/llm_evaluation/model/foxbrain_v1.2/Zh_llama2_13b_Foxbrain_Beta_v1_17_epc'
        elif version == "v7":
            model_dir = '/data/rick/pretrained_weights/LLaMA2/Zh_llama2_13b_Foxbrain_merge_Beta_1_98_epc'
        elif version == "v8":
            model_dir = '/data/rick/pretrained_weights/LLaMA2/Zh_llama2_13b_Foxbrain_merge_beta_1_98_extend_7k_tw490_slimorca'
        elif version == "v9":
            model_dir = '/data/rick/pretrained_weights/LLaMA2/Zh_llama2_13b_Foxbrain_merge_Beta_1_98_epc_continue_7k_tw490_slimorca'
        elif version == "v10":
            model_dir = "/data/rick/pretrained_weights/LLaMA2/Zh_llama2_13b_Foxbrain_merge_beta_1_98_extend_3k_tw490_slimorca"
        elif version == "v11":
            model_dir = "/data/rick/pretrained_weights/LLaMA2/Zh_Twllama_2_13b_Foxbrain_merge_beta_1_98_extend_8k_tw490_slimorca"
        else:
            raise NotImplementedError
        
        # Load model
        destroy_model_parallel()
        self.model = LLM(model=model_dir, dtype='bfloat16', gpu_memory_utilization= 0.9, swap_space = 0)
        #self.sampling_params = SamplingParams(temperature=0.4, top_p=0.98,  top_k=20, frequency_penalty=1.1, max_tokens=4096, stop_token_ids= [0])
        self.sampling_params = SamplingParams(temperature=kwargs.get('temperature',0.4), 
                                              max_tokens=kwargs.get('max_new_tokens',2048), 
                                              top_p=0.98,  
                                              top_k=20, 
                                              frequency_penalty=1.1,
                                              stop_token_ids=[0])
        
    def get_response(self, input_text: str, **kwargs) -> Dict[str, any]:

        try:
            outputs = self.model.generate(input_text, self.sampling_params)
            for output in outputs:
                generated_text = output.outputs[0].text
            #print('---------------------output_text---------------------')
            #print(generated_text)
        except:
            generated_text = ''

        return generated_text


class twllama(ResponseModel):
    def __init__(self,
                version: str = None, 
                 **kwargs) -> None:
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if version == "twllama_v1":
            model_dir = '/data1/user_guanru/llm_evaluation/model/Taiwan-LLaMa-v1.0'
        elif version == "twllama_v2":
            model_dir = '/data1/user_guanru/llm_evaluation/model/Taiwan-LLM-7B-v2.0-chat'
        elif version == "twllama_v0":
            model_dir = '/data1/user_guanru/llm_evaluation/model/Taiwan-LLaMa-v0.0'
        else:
            raise NotImplementedError

        # Load model
        destroy_model_parallel()
        self.model = LLM(model=model_dir)
        self.sampling_params = SamplingParams(temperature=0.4, top_p=0.98,  top_k=20, frequency_penalty=1.1, max_tokens=6000, stop_token_ids= [0])
        self.version = version
        
    def get_response(self, input_text: str, **kwargs) -> Dict[str, any]:
        if 'v2' in self.version:
            input_to_llm = f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {input_text} ASSISTANT:"
        else:
            input_to_llm = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {input_text} ASSISTANT:"
        print('---------------------input_text---------------------\n',input_text)
        
        try:
            outputs = self.model.generate(input_to_llm, self.sampling_params)
            for output in outputs:
                generated_text = output.outputs[0].text
            print('---------------------output_text---------------------')
            print(generated_text)
        except Exception as e: 
            print(e)
            generated_text = ''

        return generated_text