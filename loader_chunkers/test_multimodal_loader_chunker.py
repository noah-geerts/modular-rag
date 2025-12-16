import unittest
import os
import dotenv
from unittest.mock import Mock, patch, MagicMock
from loader_chunkers.multimodal_loader_chunker import MultiModalLoaderChunker
from rag_types.chunk import Content
from pathlib import Path

dotenv.load_dotenv()

class TestMultiModalLoaderChunker(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.openai_api_key = os.environ['OPENAI_API_KEY']
    cls.chunker = MultiModalLoaderChunker(openai_api_key=cls.openai_api_key)

  def test_generate_ai_summary_with_valid_content(self):
    # Arrange: Create a valid content dict
    content: Content = {
      'text': 'This is a sample document about machine learning. It covers neural networks, deep learning, and supervised learning techniques.',
      'tables': ['Table 1: ML Algorithms\nAlgorithm | Type | Accuracy\nRandom Forest | Supervised | 95%'],
      'images': []  # Empty images for this test
    }

    # Act: Generate summary
    summary = self.chunker.generate_ai_summary(content)

    # Assert: Check we got a valid string response
    self.assertIsInstance(summary, str)
    self.assertGreater(len(summary), 0)
    
    # Print first 100 characters
    print(f"\nAI Summary (first 100 chars): {summary[:100]}...")

  def test_generate_ai_summary_with_invalid_content_raises_key_error(self):
    # Arrange: Create an invalid content dict (missing 'text' key)
    invalid_content = {
      'tables': ['Table 1: Data'],
      'images': []
    }

    # Act & Assert: Should raise KeyError
    with self.assertRaises(KeyError):
      self.chunker.generate_ai_summary(invalid_content)

  # Tests for load()
  @patch('loader_chunkers.multimodal_loader_chunker.partition')
  def test_load_pdf_sets_correct_kwargs(self, mock_partition):
    # Arrange
    mock_partition.return_value = []
    pdf_file = Path("test.pdf")

    # Act
    self.chunker.load(pdf_file)

    # Assert: Verify partition was called with PDF-specific kwargs
    call_args = mock_partition.call_args
    self.assertIn('strategy', call_args.kwargs)
    self.assertEqual(call_args.kwargs['strategy'], 'hi_res')
    self.assertTrue(call_args.kwargs['extract_images_in_pdf'])
    self.assertTrue(call_args.kwargs['infer_table_structure'])

  @patch('loader_chunkers.multimodal_loader_chunker.partition')
  def test_load_txt_sets_correct_kwargs(self, mock_partition):
    # Arrange
    mock_partition.return_value = []
    txt_file = Path("test.txt")

    # Act
    self.chunker.load(txt_file)

    # Assert: Verify partition was called with TXT-specific kwargs
    call_args = mock_partition.call_args
    self.assertEqual(call_args.kwargs['strategy'], 'fast')

  @patch('loader_chunkers.multimodal_loader_chunker.partition')
  def test_load_returns_elements(self, mock_partition):
    # Arrange
    mock_elements = [Mock(), Mock()]
    mock_partition.return_value = mock_elements
    file = Path("test.pdf")

    # Act
    result = self.chunker.load(file)

    # Assert
    self.assertEqual(result, mock_elements)

  def test_load_throws_on_invalid_filetype(self):
    # Arrange
    file = Path("test.xyz")

    # Act & Assert
    with self.assertRaises(RuntimeError):
      self.chunker.load(file)

  # Tests for extract_chunk_contents()
  def test_extract_chunk_contents_text_only(self):
    # Arrange: Create mock element with only text, no orig_elements
    mock_element = Mock()
    mock_element.text = "Sample text content"
    mock_element.metadata = Mock(orig_elements=None)

    # Act
    result = self.chunker.extract_chunk_contents([mock_element])

    # Assert
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0]['text'], "Sample text content")
    self.assertEqual(result[0]['tables'], [])
    self.assertEqual(result[0]['images'], [])

  def test_extract_chunk_contents_with_no_metadata(self):
    # Arrange: Create mock element without metadata
    mock_element = Mock()
    mock_element.text = "Text without metadata"
    # No metadata attribute
    del mock_element.metadata

    # Act
    result = self.chunker.extract_chunk_contents([mock_element])

    # Assert
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0]['text'], "Text without metadata")
    self.assertEqual(result[0]['tables'], [])
    self.assertEqual(result[0]['images'], [])

  def test_extract_chunk_contents_with_tables(self):
    # Arrange: Create mock element with table
    mock_table = Mock()
    mock_table.__class__.__name__ = 'Table'
    mock_table.metadata = Mock(text_as_html="<table>HTML Table</table>")
    
    mock_element = Mock()
    mock_element.text = "Text with table"
    mock_element.metadata = Mock(orig_elements=[mock_table])

    # Act
    result = self.chunker.extract_chunk_contents([mock_element])

    # Assert
    self.assertEqual(len(result[0]['tables']), 1)
    self.assertEqual(result[0]['tables'][0], "<table>HTML Table</table>")

  def test_extract_chunk_contents_multiple_elements(self):
    # Arrange: Create multiple mock elements
    mock_element1 = Mock()
    mock_element1.text = "Text 1"
    mock_element1.metadata = Mock(orig_elements=None)

    mock_element2 = Mock()
    mock_element2.text = "Text 2"
    mock_element2.metadata = Mock(orig_elements=None)

    # Act
    result = self.chunker.extract_chunk_contents([mock_element1, mock_element2])

    # Assert
    self.assertEqual(len(result), 2)
    self.assertEqual(result[0]['text'], "Text 1")
    self.assertEqual(result[1]['text'], "Text 2")

  # Tests for create_chunks()
  def test_create_chunks_text_only(self):
    # Arrange: Content with text only (no images/tables)
    content: Content = {
      'text': 'Just plain text',
      'tables': [],
      'images': []
    }

    # Act
    result = self.chunker.create_chunks([content])

    # Assert
    self.assertEqual(len(result), 1)
    self.assertEqual(result[0]['search_text'], 'Just plain text')
    self.assertIn('content', result[0])

  @patch.object(MultiModalLoaderChunker, 'generate_ai_summary')
  def test_create_chunks_multimodal_calls_ai_summary(self, mock_ai_summary):
    # Arrange
    mock_ai_summary.return_value = "AI generated summary"
    content: Content = {
      'text': 'Text with content',
      'tables': ['Table data'],
      'images': []
    }

    # Act
    result = self.chunker.create_chunks([content])

    # Assert: Verify AI summary was called
    mock_ai_summary.assert_called_once_with(content)
    self.assertEqual(result[0]['search_text'], 'AI generated summary')

  def test_create_chunks_structure(self):
    # Arrange
    content: Content = {
      'text': 'Sample text',
      'tables': [],
      'images': []
    }

    # Act
    result = self.chunker.create_chunks([content])

    # Assert: Verify chunk has correct structure
    self.assertIsInstance(result[0], dict)
    self.assertIn('search_text', result[0])
    self.assertIn('content', result[0])

  def test_create_chunks_multiple_contents(self):
    # Arrange: Multiple contents
    content1: Content = {'text': 'Text 1', 'tables': [], 'images': []}
    content2: Content = {'text': 'Text 2', 'tables': [], 'images': []}

    # Act
    result = self.chunker.create_chunks([content1, content2])

    # Assert
    self.assertEqual(len(result), 2)
    self.assertEqual(result[0]['search_text'], 'Text 1')
    self.assertEqual(result[1]['search_text'], 'Text 2')

if __name__ == "__main__":
  unittest.main()