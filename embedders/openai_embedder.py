from embedders.embedder import Embedder
from typing import List
from openai import OpenAI
from rag_types.chunk import Chunk

class OpenAIEmbedder(Embedder):
  def __init__(self, openai_api_key):
    self.client = OpenAI(api_key=openai_api_key)

  def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
    if len(chunks) == 0:
      return []
    
    search_texts = [chunk['search_text'] for chunk in chunks]
    response = self.client.embeddings.create(model="text-embedding-3-large", input=search_texts)
    vectors = [data.embedding for data in response.data]

    return vectors