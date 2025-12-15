from abc import ABC, abstractmethod
from typing import List, Set
from langchain_core.documents import Document

class LoaderChunker(ABC):
  @property
  @abstractmethod
  def supported_extensions(self) -> Set[str]:
      pass

  @abstractmethod
  def load_and_chunk(self, path: str) -> List[Document]:
    pass