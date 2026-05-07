from fastapi import APIRouter, Request
from src.schemas.requests import GenerateRequest, GenerateResponse
import time



router = APIRouter()

# Generate endpoint
@router.post("/generate")
async def generate(request: Request, body: GenerateRequest):
    prompt = body.prompt
    max_tokens = body.max_tokens
    temperature = body.temperature

    engine = request.app.state.engine

    start_time = time.time()

    # call the engine generate
    output = await engine.generate(prompt, max_tokens, temperature)

    latency_ms = (time.time() - start_time) * 1000
    tokens_per_sec = output.get("completion_tokens") / (latency_ms / 1000)

    return GenerateResponse(
        text=output.get("output"),
        latency_ms=latency_ms,
        prompt_tokens=output.get("prompt_tokens"),
        completion_tokens=output.get("completion_tokens"),
        tokens_per_sec = tokens_per_sec
    )
