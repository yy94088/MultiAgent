import torch
from typing import Optional
from AgentPrune.llm.llm import LLM
from class_registry import ClassRegistry

class LLMRegistry:
    registry = ClassRegistry()
    _last_used_gpu = 0
    _instance_map = {}  # Global mapping table: (model_name, agent_id) -> LLM instance
    
    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None, agent_id: int = 0) -> LLM:
        """
        Create or retrieve a cached LLM instance for the given model and agent_id.
        Each unique (model_name, agent_id) pair gets its own dedicated instance that persists across batches.
        """
        # Create a unique cache key for this node
        cache_key = (model_name, agent_id)
        
        # Check if instance already exists in the mapping table
        if cache_key in cls._instance_map:
            return cls._instance_map[cache_key]
        
        # If not exists, create a new instance
        gpu_id = cls._select_optimal_gpu()
        device = f"cuda:{gpu_id}"
        model = cls.registry.get('Qwen3', model_name, device=device)
        
        # Store the instance in the mapping table
        cls._instance_map[cache_key] = model
        
        return model
    
    @classmethod
    def _select_optimal_gpu(cls, min_memory_mb: int = 1024) -> int:
            
        best_gpu = -1
        max_free = 0
        num_gpus = torch.cuda.device_count()
        
        # From the last used GPU, find the next with most free memory
        start_idx = (cls._last_used_gpu + 1) % num_gpus if cls._last_used_gpu >= 0 else 0
        
        for offset in range(num_gpus):
            i = (start_idx + offset) % num_gpus
            
            free_mb = torch.cuda.mem_get_info(i)[0] / (1024 * 1024)
            
            if free_mb >= min_memory_mb and free_mb > max_free:
                max_free = free_mb
                best_gpu = i
                
        if best_gpu >= 0:
            cls._last_used_gpu = best_gpu  # Update last used record
            
        return best_gpu if max_free > 0 else -1