from abc import ABC, abstractmethod


class BaseEngine(ABC):

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        pass