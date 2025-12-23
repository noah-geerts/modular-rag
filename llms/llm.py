from abc import ABC, abstractmethod
from typing import List

class LLM(ABC):
  # Creates completion with optional system messages and images. Throws a runtime error if the completion fails for ANY reason
  @abstractmethod
  def create_completion(self, prompt: str, system_message: str | None = None, images_base64: List[str] | None = None):
    pass