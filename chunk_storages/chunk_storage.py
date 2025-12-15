from abc import ABC, abstractmethod
from typing import List
from rag_types.chunk import Chunk

class ChunkStorage(ABC):
  @abstractmethod
  # Stores a list of chunks by id in some form of persistent storage, returning the id's as a list
  def store_chunks(self, chunks: List[Chunk]) -> List[int]:
    pass

  @abstractmethod
  # Returns a list of langchain chunks corresponding to the provided id's
  def retrieve_chunks(self, ids: List[int]) -> List[Chunk]:
    pass