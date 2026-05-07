from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct" # "gpt2" # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: str = "cuda"
    max_tokens: int = 256
    temperature: float = 0.5
    engine_type: str = "vllm" # hf or vllm


settings = Settings()