"""
Multi-GPU Transformer Model Pool Implementation

This module implements a model pool architecture that manages multiple transformer
model instances across multiple GPUs, enabling higher throughput for concurrent
inference workloads.

Architecture:
    - Maintains N model instances distributed across available GPUs
    - Round-robin scheduling for load balancing
    - Thread-safe access with individual locks per model instance
    
"""

import torch
from typing import List, Union, Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import functools
import time
import logging
import gc

from AgentPrune.llm.format import Message
from AgentPrune.llm.llm import LLM
from AgentPrune.llm.llm_registry import LLMRegistry

logger = logging.getLogger(__name__)


@LLMRegistry.register('MultiGPUTransformers')
class MultiGPUTransformers(LLM):
    
    DEFAULT_MAX_TOKENS = 2048
    DEFAULT_TEMPERATURE = 0.7
    DEFUALT_NUM_COMPLETIONS = 1
    
    def __init__(
        self, 
        model_path: str, 
        num_instances: int = 5,
        device_ids: List[int] = None
    ) -> None:
        """
        Initialize multi-instance model pool with GPU distribution.
        
        Args:
            model_path: Absolute or relative path to model directory
            num_instances: Number of model replicas (default: 5)
            device_ids: GPU device IDs for placement (default: [0, 1])
        """
        self.model_path = str(Path(model_path).resolve())
        self.num_instances = num_instances
        
        # Default to using 2 GPUs if not specified
        if device_ids is None:
            device_ids = [0, 1]
        self.device_ids = device_ids
        
        # Store original device_ids for deepcopy
        self._original_device_ids = device_ids
        
        # Validate GPU availability
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            raise RuntimeError("No CUDA devices available for multi-GPU inference")
        
        for device_id in device_ids:
            if device_id >= available_gpus:
                raise ValueError(
                    f"Device ID {device_id} exceeds available GPUs ({available_gpus})"
                )
        
        # Display GPU memory information
        logger.info("GPU Memory Status:")
        for device_id in set(device_ids):
            total_mem = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            free_mem = total_mem - allocated
            logger.info(f"  GPU {device_id}: {free_mem:.2f}GB free / {total_mem:.2f}GB total")
        
        # Estimate memory requirements
        instances_per_gpu = {}
        for i in range(num_instances):
            gpu_id = device_ids[i % len(device_ids)]
            instances_per_gpu[gpu_id] = instances_per_gpu.get(gpu_id, 0) + 1
        
        logger.info(f"Planned distribution:")
        for gpu_id, count in sorted(instances_per_gpu.items()):
            logger.info(f"  GPU {gpu_id}: {count} instances (~{count * 16}GB estimated)")
        
        logger.info(f"Initializing {num_instances} model instances across GPUs {device_ids}")
        
        # Load tokenizer once (shared across all instances)
        logger.info(f"Loading tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Check model size from config
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            if hasattr(config, 'num_parameters'):
                num_params = config.num_parameters
            elif hasattr(config, 'n_params'):
                num_params = config.n_params
            else:
                # Estimate from hidden_size and num_layers
                num_params = getattr(config, 'hidden_size', 0) * getattr(config, 'num_hidden_layers', 0) * 12
            
            num_params_b = num_params / 1e9 if num_params > 0 else 8.0
            logger.info(f"Model size: ~{num_params_b:.1f}B parameters")
            
            # Warn if model is larger than expected
            if num_params_b > 20:
                logger.warning(f"⚠️  Model appears to be {num_params_b:.1f}B parameters (not 8B!)")
                logger.warning(f"    Each instance may require {num_params_b * 2:.0f}GB+ GPU memory")
                logger.warning(f"    Consider reducing --num_instances significantly")
        except Exception as e:
            logger.warning(f"Could not determine model size: {e}")
        
        # Initialize model instances with round-robin GPU placement
        self.model_instances = []
        self.locks = []
        
        for i in range(num_instances):
            # Distribute models across GPUs in round-robin fashion
            device_id = device_ids[i % len(device_ids)]
            
            logger.info(f"Loading model instance {i+1}/{num_instances} on GPU {device_id}")
            
            # Clear CUDA cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            try:
                # Calculate max memory per GPU (leave 5GB buffer for safety)
                gpu_total_memory = torch.cuda.get_device_properties(device_id).total_memory
                instances_on_this_gpu = device_ids[:i+1].count(device_id)
                max_mem_per_instance = (gpu_total_memory * 0.9) / max(1, instances_on_this_gpu + device_ids[i+1:].count(device_id))
                max_memory = {device_id: f"{int(max_mem_per_instance / 1024**3)}GiB"}
                
                logger.debug(f"Max memory for instance {i+1} on GPU {device_id}: {max_memory}")
                
                # Load model with strict memory limit
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",  # Let transformers handle placement
                    max_memory=max_memory,  # Enforce memory limit
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    offload_folder=None  # Don't use disk offloading
                )
                
                # Manually move to target device if needed
                if next(model.parameters()).device.index != device_id:
                    logger.warning(f"Model not on target GPU {device_id}, moving...")
                    model = model.to(f"cuda:{device_id}")
                
                model.eval()
                
                self.model_instances.append(model)
                self.locks.append(threading.Lock())
                
                # Force synchronization and cleanup after each load
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device_id)
                    torch.cuda.empty_cache()
                
                # Log memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
                    logger.info(f"Instance {i+1} loaded on GPU {device_id}: "
                              f"{allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                else:
                    logger.info(f"Instance {i+1} loaded successfully on GPU {device_id}")
                    
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"Failed to load instance {i+1} on GPU {device_id}: {e}")
                
                # Show memory info for all GPUs
                logger.error(f"\n⚠️  CUDA Out of Memory Error")
                logger.error(f"\nCurrent GPU Memory Status:")
                for gid in set(device_ids):
                    total = torch.cuda.get_device_properties(gid).total_memory / 1024**3
                    allocated = torch.cuda.memory_allocated(gid) / 1024**3
                    reserved = torch.cuda.memory_reserved(gid) / 1024**3
                    free = total - allocated
                    logger.error(f"  GPU {gid}: {allocated:.1f}GB used / {total:.1f}GB total ({free:.1f}GB free)")
                
                logger.error(f"\nConfiguration:")
                logger.error(f"  - Model path: {self.model_path}")
                logger.error(f"  - Instances requested: {num_instances}")
                logger.error(f"  - Instances loaded successfully: {i}")
                logger.error(f"  - Device distribution: {device_ids}")
                
                logger.error(f"\n🔧 Solutions:")
                if i == 0:
                    logger.error(f"  ❌ Cannot load even 1 instance! This model is TOO LARGE.")
                    logger.error(f"     Check if this is really Qwen3-8B (8 billion params).")
                    logger.error(f"     The model might be Qwen3-72B or larger.")
                    logger.error(f"     Each 8B model should use ~16-20GB, not 79GB!")
                elif i == 1:
                    logger.error(f"  ✅ Use --num_instances 1 (single instance mode)")
                    logger.error(f"  ✅ Or use separate GPUs: --device_ids '[0]' or '[1]'")
                else:
                    logger.error(f"  ✅ Reduce to --num_instances {i}")
                    logger.error(f"  ✅ Balance better: --device_ids {[gid for j in range(i) for gid in [device_ids[j % len(set(device_ids))]]][:i]}")
                
                raise RuntimeError(
                    f"CUDA OOM: Successfully loaded {i}/{num_instances} instances. "
                    f"Cannot fit more on GPU {device_id}. Try --num_instances {max(1, i)}"
                ) from e
        
        # Round-robin counter for load balancing
        self.current_idx = 0
        self._idx_lock = threading.Lock()
        
        # Dedicated thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=num_instances * 2)
        
        logger.info(f"Multi-GPU model pool initialized with {num_instances} instances")
    
    def __deepcopy__(self, memo: Dict) -> 'MultiGPUTransformers':
        """
        Custom deepcopy to avoid copying model parameters.
        
        When copying the MultiGPUTransformers object (e.g., during graph.deepcopy()),
        we don't want to duplicate the actual model weights. Instead, we return
        a new object that references the same shared model instances.
        
        This prevents CUDA OOM errors that would occur if deepcopy tried to
        clone all model parameters to GPU memory.
        
        Returns:
            A new MultiGPUTransformers instance sharing the same models
            
        Note:
            This is critical for AgentPrune's workflow which calls deepcopy(graph)
            in each training iteration. Without this, each deepcopy would try to
            allocate another 80GB+ of GPU memory.
        """
        # Create a new instance without calling __init__
        cls = self.__class__
        new_obj = cls.__new__(cls)
        
        # Copy only the metadata (not the model instances)
        new_obj.model_path = self.model_path
        new_obj.num_instances = self.num_instances
        new_obj.device_ids = self._original_device_ids
        new_obj._original_device_ids = self._original_device_ids
        
        # Share the tokenizer (lightweight)
        new_obj.tokenizer = self.tokenizer
        
        # Share the model instances (avoid duplication)
        new_obj.model_instances = self.model_instances
        new_obj.locks = self.locks
        
        # Share the thread pool and counter
        new_obj._executor = self._executor
        new_obj.current_idx = self.current_idx
        new_obj._idx_lock = self._idx_lock
        
        logger.debug(f"Deepcopy of MultiGPUTransformers: sharing {len(self.model_instances)} model instances")
        
        return new_obj
    
    def _get_next_instance(self) -> int:
        """
        Get next available model instance using round-robin scheduling.
        
        Returns:
            Index of the selected model instance
            
        Note:
            This implements a simple round-robin strategy. More sophisticated
            scheduling (e.g., least-loaded, priority-based) can be added.
        """
        with self._idx_lock:
            idx = self.current_idx
            self.current_idx = (self.current_idx + 1) % self.num_instances
            return idx
    
    def _format_messages(self, messages: Union[str, List[Message]]) -> str:
        """
        Format conversation messages into model-specific prompt format.
        
        Args:
            messages: List of conversation messages or raw string
            
        Returns:
            Formatted prompt string
        """
        if isinstance(messages, str):
            return messages
        
        # Try using chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                msg_dicts = []
                for msg in messages:
                    if isinstance(msg, dict):
                        msg_dicts.append(msg)
                    else:
                        msg_dicts.append({'role': msg.role, 'content': msg.content})
                
                formatted = self.tokenizer.apply_chat_template(
                    msg_dicts,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted
            except Exception as e:
                logger.warning(f"Chat template failed, using fallback: {e}")
        
        # Fallback format
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
    ) -> Union[str, List[str]]:
        """
        Generate text using an available model instance from the pool.
        
        This method selects a model instance using round-robin scheduling and
        performs inference with thread-safe access control.
        
        Args:
            messages: Input conversation or prompt
            max_tokens: Maximum tokens to generate (default: 2048)
            temperature: Sampling temperature (default: 0.7)
            num_comps: Number of completions (default: 1)
            
        Returns:
            Generated text string or list of strings
        """
        start_time = time.time()
        
        # Get parameters
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        # Select model instance with round-robin
        instance_idx = self._get_next_instance()
        model = self.model_instances[instance_idx]
        lock = self.locks[instance_idx]
        
        device = next(model.parameters()).device
        logger.debug(f"Using model instance {instance_idx} on {device}")
        
        # Format prompt
        prompt = self._format_messages(messages)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
        
        # Move to correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logger.debug(f"[Instance {instance_idx}] Input length: {inputs['input_ids'].shape[1]} tokens")
        
        # Thread-safe generation
        with lock:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    num_return_sequences=num_comps,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        generation_time = time.time() - start_time
        logger.debug(f"[Instance {instance_idx}] Generation completed in {generation_time:.2f}s")
        
        # Decode output
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, input_length:]
        
        responses = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        if num_comps == 1:
            return responses[0].strip()
        else:
            return [r.strip() for r in responses]
    
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """
        Asynchronous text generation interface.
        
        Enables non-blocking inference for concurrent agent operations.
        
        Args:
            messages: Input conversation or prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_comps: Number of completions
            
        Returns:
            Generated text string or list of strings
        """
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
        """
        Synchronous text generation interface.
        
        Args:
            messages: Input conversation or prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_comps: Number of completions
            
        Returns:
            Generated text string or list of strings
        """
        return self._generate(messages, max_tokens, temperature, num_comps)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the model pool.
        
        Returns:
            Dictionary with pool information:
                - num_instances: Number of model instances
                - device_ids: GPU devices in use
                - memory_per_gpu: Memory allocated per GPU
        """
        stats = {
            'num_instances': self.num_instances,
            'device_ids': self.device_ids,
            'instances_per_device': {}
        }
        
        for i, model in enumerate(self.model_instances):
            device = next(model.parameters()).device
            device_str = str(device)
            
            if device_str not in stats['instances_per_device']:
                stats['instances_per_device'][device_str] = []
            
            stats['instances_per_device'][device_str].append(i)
        
        # Get memory stats if on CUDA
        if torch.cuda.is_available():
            stats['memory_per_gpu'] = {}
            for device_id in self.device_ids:
                allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(device_id) / 1024**3  # GB
                stats['memory_per_gpu'][f'cuda:{device_id}'] = {
                    'allocated_gb': f"{allocated:.2f}",
                    'reserved_gb': f"{reserved:.2f}"
                }
        
        return stats
