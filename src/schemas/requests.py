from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.5


class GenerateResponse(BaseModel):
    text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    tokens_per_sec: float