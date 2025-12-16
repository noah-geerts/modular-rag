from typing import List
import unittest
from embedders.openai_embedder import OpenAIEmbedder
import dotenv
import os

from rag_types.chunk import Chunk

dotenv.load_dotenv()

class TestOpenAIEmbedder(unittest.TestCase):
  def test_embedding_3_chunks(self):
    # Arrange
    toEmbed: List[Chunk] = [
      {"search_text": "a", "metadata": {"original_content": {"image": "niceImageA", "table": "coolTableA"}}},
      {"search_text": "b", "metadata": {"original_content": {"image": "niceImageB", "table": "coolTableB"}}},
      {"search_text": "c", "metadata": {"original_content": {"image": "niceImageC", "table": "coolTableC"}}}
    ]
    embedder = OpenAIEmbedder(openai_api_key=os.environ['OPENAI_API_KEY'])

    # Act & Assert
    vectors = embedder.embed_chunks(toEmbed)
    self.assertEqual(len(vectors), 3)
    self.assertEqual(len(vectors[0]), 3072)

  def test_embedding_0_chunks(self):
    # Arrange
    toEmbed = []
    embedder = OpenAIEmbedder(openai_api_key=os.environ['OPENAI_API_KEY'])

    # Act & Assert
    vectors = embedder.embed_chunks(toEmbed)
    self.assertEqual(len(vectors), 0)

if __name__ == "__main__":
  unittest.main()