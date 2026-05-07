import uuid
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from src.engine.base import BaseEngine



class VLLMEngine(BaseEngine):

    def __init__(self, model_name: str):

        self.model_name = model_name

        # Load the model (similar to HF but optimized backend)
        engine_args = AsyncEngineArgs(
            model=model_name,
            dtype="float16"
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args) 

    async def generate(self, prompt: str, max_tokens: int, temperature: float):

        # Define sampling params (equivalent to generate() args)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens
        )

        # Each request needs a unique request_id
        request_id = str(uuid.uuid4())

        # Async generator (streaming internally)
        result_generator = self.engine.generate(prompt, sampling_params, request_id)

        final_output = None

        async for output in result_generator:
            final_output = output # keep last chunk (final result)

        
        # Extract response
        response_text = final_output.outputs[0].text

        # Token usage
        prompt_tokens_length = len(final_output.prompt_token_ids)
        completion_tokens_length = len(final_output.outputs[0].token_ids) 


        return {
            "output": response_text,
            "prompt_tokens": prompt_tokens_length,
            "completion_tokens": completion_tokens_length
        }


