from typing import TypedDict, List

class Content(TypedDict):
    """
    Represents a content object containing text, tables, and images.
    """
    text: str
    tables: List[str]
    images: List[str]

class Chunk(TypedDict):
  search_text: str
  content: Content