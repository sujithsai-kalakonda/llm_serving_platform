from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    temperature: float

class GenerateResponse(BaseModel):
    text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    tokens_per_sec: float