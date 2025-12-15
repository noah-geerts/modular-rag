from chunk_storages.chunk_storage import ChunkStorage
from typing import List
from rag_types.chunk import Chunk
import sqlite3
import json

class SQLiteChunkStorage(ChunkStorage):

  def __init__(self, db_name: str, table_name: str):
    if not db_name:
      raise RuntimeError("SQLiteChunkStorage requires a db_name.")
    if not table_name:
      raise RuntimeError("SQLiteChunkStorage requires a table_name.")

    self.db_name = db_name
    self.table_name = table_name

    self.conn = sqlite3.connect(self.db_name)
    self.cur = self.conn.cursor()

    existing_table = self.cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (self.table_name,)
    ).fetchone()

    if existing_table is None:
      self.cur.execute(
        f"CREATE TABLE {self.table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT, chunk_json TEXT)"
      )
      self.conn.commit()

  def store_chunks(self, chunks: List[Chunk]) -> List[int]:
    ids = []
    for chunk in chunks:
      chunk_json = json.dumps(chunk)
      self.cur.execute(f"INSERT INTO {self.table_name} (chunk_json) VALUES (?) RETURNING id", (chunk_json,))
      ids.append(self.cur.fetchone()[0])
    self.conn.commit()
    return ids
  
  def retrieve_chunks(self, ids: List[int]) -> List[Chunk]:
    if not ids:
      return []
    id_placeholders = ",".join("?" for _ in ids)
    self.cur.execute(f"SELECT chunk_json FROM {self.table_name} WHERE id IN ({id_placeholders})", ids)
    rows = self.cur.fetchall()
    chunks = [json.loads(row[0]) for row in rows]
    return chunks