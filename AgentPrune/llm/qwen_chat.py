import os
import copy
import torch
import pynvml
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Union, Optional
from AgentPrune.llm.llm import LLM
from AgentPrune.llm.llm_registry import LLMRegistry

def get_best_gpu() -> int:
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        best_gpu_index = 0
        max_free_memory = 0
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            if info.free > max_free_memory:
                max_free_memory = info.free
                best_gpu_index = i
        
        return best_gpu_index
    except Exception as e:
        return 0

@LLMRegistry.register('local')
class QwenChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        target_gpu = get_best_gpu()
        self.device = torch.device(f"cuda:{target_gpu}")
        self.abspath = os.path.join(os.path.dirname(__file__),"/home/wangchichu/Qwen/",model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.abspath, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" 
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.abspath,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map={"": self.device}, 
            trust_remote_code=True
        ).eval()

    def _format_messages(self, messages: List[Dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    async def gen_batch(
        self, 
        batch_messages: List[List[Dict]], 
        max_tokens: int = 512, 
        temperature: float = 0.7
    ) -> List[str]:
        if not batch_messages:
            return []

        prompts = [self._format_messages(m) for m in batch_messages]
        
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=self.tokenizer.pad_token_id
        )

        responses = []
        for i in range(len(prompts)):
            input_len = inputs.input_ids[i].shape[0]
            gen_text = self.tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True)
            responses.append(gen_text.strip())
            
        return responses

    async def agen(self, messages: List[Dict], **kwargs) -> str:
        results = await self.gen_batch([messages], **kwargs)
        return results[0]

    def gen(self, messages: List[Dict], **kwargs) -> str:
        
        return asyncio.run(self.agen(messages, **kwargs))
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        
        new_obj = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_obj

        new_obj.model_name = self.model_name
        
        new_obj.model = self.model
        new_obj.tokenizer = self.tokenizer
        new_obj.device = self.device
        
        return new_obj