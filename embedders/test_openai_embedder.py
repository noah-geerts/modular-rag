import unittest
from openai_embedder import OpenAIEmbedder
from langchain_core.documents import Document
import dotenv
import os

dotenv.load_dotenv()

class TestOpenAIEmbedder(unittest.TestCase):
  def test_embedding_3_documents(self):
    # Arrange
    toEmbed = [
      Document(page_content="a", metadata={"original_content": {"image": "niceImageA", "table": "coolTableA"}}),
      Document(page_content="b", metadata={"original_content": {"image": "niceImageB", "table": "coolTableB"}}),
      Document(page_content="c", metadata={"original_content": {"image": "niceImageC", "table": "coolTableC"}})
    ]
    embedder = OpenAIEmbedder(openai_api_key=os.environ['OPENAI_API_KEY'])

    # Act & Assert
    vectors = embedder.embed_documents(toEmbed)
    self.assertEqual(len(vectors), 3)
    self.assertEqual(len(vectors[0]), 3072)

  def test_embedding_0_documents(self):
    # Arrange
    toEmbed = []
    embedder = OpenAIEmbedder(openai_api_key=os.environ['OPENAI_API_KEY'])

    # Act & Assert
    vectors = embedder.embed_documents(toEmbed)
    self.assertEqual(len(vectors), 0)

if __name__ == "__main__":
  unittest.main()