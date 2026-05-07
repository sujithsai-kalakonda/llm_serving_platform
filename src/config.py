from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    model_name: str = "gpt2" # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_tokens: int = 256
    temperature: float = 0.5
    engine_type: str = "vllm" # hf or vllm


settings = Settings()