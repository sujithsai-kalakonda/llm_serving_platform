from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from .base import BaseEngine

class HFEngine(BaseEngine): 

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name

        # Load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"   # auto places on GPU if available
        )

    def generate(self, prompt: str, max_tokens: int, temperature: float):
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True # do_sample: False → deterministic (same output every time, greedy decoding), True  → enables randomness (sampling from probability distribution). Needed for creative / varied responses
        )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]

        # Decode output
        output = self.tokenizer.decode(
            generated_tokens, 
            skip_special_tokens=True)

        return output

