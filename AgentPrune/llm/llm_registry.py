import torch
from typing import Optional
from AgentPrune.llm.llm import LLM
from class_registry import ClassRegistry
from AgentPrune.llm.load_transformers import LoadTransformers

class LLMRegistry:
    registry = ClassRegistry()
    _last_used_gpu = -1
    # Class-level cache for shared LLM instances
    _llm_cache = {}

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None, agent_id: int = None) -> LLM:
        """
        Get or create an LLM instance.
        If agent_id is provided, automatically assigns to available GPU.
        Otherwise, creates shared instance (cached).
        Args:
            model_name: Model identifier
            agent_id: Optional agent identifier for dedicated instance
        Returns:
            LLM instance
        """
        print("agent_id", agent_id)
        cache_key = f"{model_name}:agent_{agent_id}"
        if cache_key in cls._llm_cache:
            print("cache hit", cache_key)
            return cls._llm_cache[cache_key]
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_id = cls._select_optimal_gpu()
            device = f"cuda:{gpu_id}"
        else:
            device = "cpu"
        print("cache hit", cache_key)
        model = LoadTransformers(model_name, device=device)

        cls._llm_cache[cache_key] = model
            
        return cls._llm_cache[cache_key]
    
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