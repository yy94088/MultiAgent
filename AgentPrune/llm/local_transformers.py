"""
Local Transformer Model Implementation

This module implements a local Transformer-based language model interface using 
Hugging Face's transformers library. It provides efficient model loading with 
caching mechanisms and supports both synchronous and asynchronous inference.

"""

import torch
from typing import List, Union, Optional, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import functools
import time
import logging

from AgentPrune.llm.format import Message
from AgentPrune.llm.llm import LLM


# Configure module logger
logger = logging.getLogger(__name__)



class LocalTransformers(LLM):
    """
    Local Transformer Model Wrapper with Caching and Asynchronous Support
    
    This class implements a thread-safe, memory-efficient wrapper for locally-hosted
    transformer models. It employs a class-level caching mechanism to prevent redundant
    model loading and supports both synchronous and asynchronous inference patterns.
    
    The implementation is designed for multi-agent systems where multiple instances
    may share the same underlying model, optimizing memory usage and initialization time.
    
    Attributes:
        _model_cache (dict): Class-level cache for loaded model instances
        _tokenizer_cache (dict): Class-level cache for tokenizer instances
        _lock (threading.Lock): Thread synchronization primitive for safe model loading
        _executor (ThreadPoolExecutor): Thread pool for asynchronous execution
        model_path (str): Absolute path to the model directory
        device (str): Target device for model inference ('auto', 'cuda', 'cpu')
    
    Examples:
        >>> llm = LocalTransformers("./Qwen/Qwen3-8B")
        >>> messages = [Message(role="user", content="Explain attention mechanism")]
        >>> response = llm.gen(messages, max_tokens=100)
        
    """
    
    # Class-level shared resources for memory efficiency
    _model_cache: Dict[str, Any] = {}
    _tokenizer_cache: Dict[str, Any] = {}
    _lock: threading.Lock = threading.Lock()
    _executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=10)
    
    def __init__(self, model_path: str, device: str = "auto") -> None:
        self.model_path: str = str(Path(model_path).resolve())
        self.device: str = device
        self._ensure_model_loaded()
    
    def __deepcopy__(self, memo: Dict) -> 'LocalTransformers':
        cls = self.__class__
        new_obj = cls.__new__(cls)
        new_obj.model_path = self.model_path
        new_obj.device = self.device
        return new_obj
    
    def _ensure_model_loaded(self) -> None:
        cache_key = self.model_path

        if cache_key in self._model_cache:
            return

        with self._lock:

            if cache_key in self._model_cache:
                return
            
            logger.info(f"Loading model from: {self.model_path}")
            
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.debug("Padding token set to EOS token")
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            
            # Set to evaluation mode (disable dropout, etc.)
            model.eval()

            # Cache for reuse
            self._model_cache[cache_key] = model
            self._tokenizer_cache[cache_key] = tokenizer
            
            # Log model configuration
            device_info = next(model.parameters()).device
            dtype_info = next(model.parameters()).dtype
            logger.info(f"Model loaded successfully: device={device_info}, dtype={dtype_info}")
    
    def _get_model(self) -> Any:
        cache_key = self.model_path
        return self._model_cache[cache_key]
    
    def _get_tokenizer(self) -> Any:
        cache_key = self.model_path
        return self._tokenizer_cache[cache_key]
    
    def _format_messages(self, messages: List[Message]) -> str:
        if isinstance(messages, str):
            return messages
        
        tokenizer = self._get_tokenizer()
        
        # Attempt to use model-specific chat template
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                msg_dicts = []
                for msg in messages:
                    if isinstance(msg, dict):
                        msg_dicts.append(msg)
                    else:
                        msg_dicts.append({'role': msg.role, 'content': msg.content})
                
                formatted = tokenizer.apply_chat_template(
                    msg_dicts,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted
            except Exception as e:
                logger.warning(f"Chat template application failed, using fallback format: {e}")
        
        # Generic fallback format
        formatted_text = ""
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
            else:
                role = msg.role
                content = msg.content
            
            if role == "system":
                formatted_text += f"System: {content}\n\n"
            elif role == "user":
                formatted_text += f"User: {content}\n\n"
            elif role == "assistant":
                formatted_text += f"Assistant: {content}\n\n"
        
        formatted_text += "Assistant:"
        return formatted_text
    
    def _generate(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> str:
        start_time = time.time()
        
        model = self._get_model()
        tokenizer = self._get_tokenizer()
        
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        prompt = self._format_messages(messages)
        
        # Tokenize input with truncation to prevent memory overflow
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logger.debug(f"Input sequence length: {inputs['input_ids'].shape[1]} tokens")
        
        # Execute autoregressive generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                num_return_sequences=num_comps,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generation_time = time.time() - start_time
        logger.debug(f"Generation completed in {generation_time:.2f} seconds")
        
        # Decode only the newly generated tokens (excluding prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, input_length:]
        
        responses = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        if num_comps == 1:
            return responses[0].strip()
        else:
            return [r.strip() for r in responses]
    
    def _generate_with_hidden_states(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        return_hidden_states: bool = True,
        return_past_key_values: bool = True,
    ) -> Dict[str, any]:

        start_time = time.time()
        
        model = self._get_model()
        tokenizer = self._get_tokenizer()
        
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        prompt = self._format_messages(messages)
        
        logger.info(f"Starting generation with internal states (max_tokens={max_tokens})")
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logger.debug(f"Input sequence length: {inputs['input_ids'].shape[1]} tokens")
        
        # Generate with internal state tracking
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                num_return_sequences=num_comps,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_hidden_states=return_hidden_states,
                output_attentions=False,
                return_dict_in_generate=True,
            )
        
        generation_time = time.time() - start_time
        logger.info(f"Generation with states completed in {generation_time:.2f} seconds")
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        
        responses = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        result = {
            'text': responses[0].strip() if num_comps == 1 else [r.strip() for r in responses]
        }
        
        # Extract hidden states if available
        if return_hidden_states and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            result['hidden_states'] = outputs.hidden_states
            logger.info(f"Hidden states captured:")
            logger.info(f"  - Generation steps: {len(outputs.hidden_states)}")
            if len(outputs.hidden_states) > 0:
                logger.info(f"  - Layers per step: {len(outputs.hidden_states[0])}")
                logger.info(f"  - Final layer shape: {outputs.hidden_states[-1][-1].shape}")
        
        # Extract KV cache if available
        if return_past_key_values and hasattr(outputs, 'past_key_values') and outputs.past_key_values:
            result['past_key_values'] = outputs.past_key_values
            logger.info(f"KV cache captured:")
            logger.info(f"  - Number of layers: {len(outputs.past_key_values)}")
            if len(outputs.past_key_values) > 0:
                key_shape = outputs.past_key_values[0][0].shape
                value_shape = outputs.past_key_values[0][1].shape
                logger.info(f"  - Key tensor shape: {key_shape} [batch, num_heads, seq_len, head_dim]")
                logger.info(f"  - Value tensor shape: {value_shape}")
        
        return result
    
    async def agen_with_hidden_states(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        return_hidden_states: bool = True,
        return_past_key_values: bool = True,
    ) -> Dict[str, any]:
        loop = asyncio.get_event_loop()
        
        func = functools.partial(
            self._generate_with_hidden_states,
            messages,
            max_tokens,
            temperature,
            num_comps,
            return_hidden_states,
            return_past_key_values
        )
        
        return await loop.run_in_executor(self._executor, func)
    
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        loop = asyncio.get_event_loop()

        func = functools.partial(
            self._generate,
            messages,
            max_tokens,
            temperature,
            num_comps
        )
        
        return await loop.run_in_executor(
            self._executor,
            func
        )
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        return self._generate(messages, max_tokens, temperature, num_comps)
