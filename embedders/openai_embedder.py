from embedder import Embedder
from typing import List
from langchain_core.documents import Document
from openai import OpenAI

class OpenAIEmbedder(Embedder):
  def __init__(self, openai_api_key):
    self.client = OpenAI(api_key=openai_api_key)

  def embed_documents(self, documents: List[Document]) -> List[List[float]]:
    if len(documents) == 0:
      return []
    
    search_texts = [document.page_content for document in documents]
    response = self.client.embeddings.create(model="text-embedding-3-large", input=search_texts)
    vectors = [data.embedding for data in response.data]

    return vectors