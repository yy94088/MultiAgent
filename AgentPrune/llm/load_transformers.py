from typing import List, Union, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from AgentPrune.llm.llm import LLM
from AgentPrune.llm.format import Message


class LoadTransformers(LLM):
    """
    Loading and using HuggingFace Transformer models.
    """
    
    def __init__(self, model_name: str, device: str = "cuda:1"):
        """
        Initialize the LoadTransformers instance.
        
        Args:
            model_name: Path or name of the HuggingFace model
            device: Device to load the model on (e.g., "cpu", "cuda:0")
        """
        self.model_name = model_name
        self.device = device
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map=device if "cuda" in device else None,
            trust_remote_code=True
        )
        
        self.model.eval()
    
    def _format_messages(self, messages: List[Message]) -> str:
        """
        Convert messages to a prompt string.
        
        Args:
            messages: List of Message objects
            
        Returns:
            Formatted prompt string
        """
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant:"
        return prompt
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """
        Generate text completion(s) synchronously.
        
        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_comps: Number of completions to generate
            
        Returns:
            Generated text or list of texts
        """
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        temperature = temperature or self.DEFAULT_TEMPERATURE
        num_comps = num_comps or self.DEFUALT_NUM_COMPLETIONS
        
        # Format messages into a prompt
        prompt = self._format_messages(messages)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                num_return_sequences=num_comps,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            # Skip input tokens
            generated_ids = output[inputs['input_ids'].shape[1]:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(text.strip())
        
        return generated_texts[0] if num_comps == 1 else generated_texts
    
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """
        Generate text completion(s) asynchronously.
        
        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_comps: Number of completions to generate
            
        Returns:
            Generated text or list of texts
        """
        # For simplicity, just call the synchronous version
        # In a production setting, you might want to use asyncio.to_thread
        return self.gen(messages, max_tokens, temperature, num_comps)
