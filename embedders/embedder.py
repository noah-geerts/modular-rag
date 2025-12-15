from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class Embedder(ABC):
  @abstractmethod
  def embed_documents(self, documents: List[Document]):
    pass