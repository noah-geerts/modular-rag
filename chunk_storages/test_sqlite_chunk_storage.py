import unittest
import os
import sqlite3
from typing import List
from chunk_storages.sqlite_chunk_storage import SQLiteChunkStorage
from rag_types.chunk import Chunk

SQLITE_DB_NAME = 'test_chunks.db'
SQLITE_TABLE_NAME = 'chunks'

class TestSQLiteChunkStorage(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    # Create the SQLite database
    conn = sqlite3.connect(SQLITE_DB_NAME)
    conn.close()

  @classmethod
  def tearDownClass(cls):
    # Clean up: remove test database
    if os.path.exists(SQLITE_DB_NAME):
      os.remove(SQLITE_DB_NAME)

  def test_store_and_retrieve_chunks(self):
    # Arrange
    expectedChunks: List[Chunk] = [
      {"search_text": "a", "metadata": {"original_content": {"image": "niceImageA", "table": "coolTableA"}}},
      {"search_text": "b", "metadata": {"original_content": {"image": "niceImageB", "table": "coolTableB"}}},
      {"search_text": "c", "metadata": {"original_content": {"image": "niceImageC", "table": "coolTableC"}}}
    ]
    expectedIds = [1, 2, 3]

    # Act & Assert
    sqliteStorage = SQLiteChunkStorage(SQLITE_DB_NAME, SQLITE_TABLE_NAME)
    actualIds = sqliteStorage.store_chunks(expectedChunks)
    self.assertEqual(actualIds, expectedIds)

    actualChunks = sqliteStorage.retrieve_chunks(actualIds)
    self.assertEqual(actualChunks, expectedChunks)

  def test_store_and_retrieve_0_chunks(self):
    # Arrange
    expectedChunks = []
    expectedIds = []

    # Act & Assert
    sqliteStorage = SQLiteChunkStorage(SQLITE_DB_NAME, SQLITE_TABLE_NAME)
    actualIds = sqliteStorage.store_chunks(expectedChunks)
    self.assertEqual(actualIds, expectedIds)

    actualChunks = sqliteStorage.retrieve_chunks(actualIds)
    self.assertEqual(actualChunks, expectedChunks)


if __name__ == '__main__':
  unittest.main()