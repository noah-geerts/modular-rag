from abc import ABC, abstractmethod
from typing import List

class Embedder(ABC):
  @abstractmethod
  def embed_strings(self, strings: List[str]) -> List[List[float]]:
    pass