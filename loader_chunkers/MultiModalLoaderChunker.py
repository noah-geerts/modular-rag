import json
import pickle
from typing import List, Set, TypedDict
from langchain_core.documents import Document
from .LoaderChunker import LoaderChunker
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from pathlib import Path
from langchain_core.messages import HumanMessage

class Content(TypedDict):
    """
    Represents a content object containing text, tables, and images.
    """
    text: str
    tables: List[str]
    images: List[str] 

class MultiModalLoaderChunker(LoaderChunker):
  @property
  def supported_extensions(self) -> Set[str]:
    return {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".txt", ".md", ".pdf"}
  
  def load(self, file: Path) -> List[Element]:
    suffix = file.suffix.lower()

    kwargs = {"filename": str(file)}

    if suffix == ".pdf":
      kwargs |= {
        "strategy": "hi_res",  # High accuracy OCR and layout detection
        "extract_images_in_pdf": True,  # Extract images from PDF
        "infer_table_structure": True,  # Detect and preserve table structure
        "extract_image_block_types": ["Image", "Table"],  # Include images and tables
      }
    elif suffix in [".png", ".jpg", ".jpeg"]:
      kwargs |= {
        "strategy": "hi_res",  # High accuracy OCR for images
        "infer_table_structure": True,  # Detect tables in images
      }
    elif suffix == ".docx":
      kwargs |= {
        "infer_table_structure": True,  # Preserve table structure
      }
    elif suffix in [".txt", ".md"]:
      kwargs |= {
        "strategy": "fast",  # Plain text doesn't need hi_res
      }

    return partition(**kwargs)
  
  # Extracts images and tables from CompositeElement chunks into custom dict
  def separate_content_types(self, elementChunk: Element):
    content: Content = {
      'text': elementChunk.text,
      'tables': [],
      'images': []
    }

    # Ensure the provided elementChunk has the correct fields
    if hasattr(elementChunk, 'metadata') and hasattr(elementChunk.metadata, 'orig_elements'):
      for element in elementChunk.metadata.orig_elements:
        element_type = type(element).__name__
        print("      " + element_type)

        # Tables
        if element_type == 'Table':
          table_html = getattr(element.metadata, 'text_as_html', element.text)
          content['tables'].append(table_html)

        # Images
        elif element_type == 'Image':
          if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
            content['images'].append(element.metadata.image_base64)

      return content
    
  # Creates an AI summary of a chunk containing images and tables that it can be searched by
  # (we will still return the original images, tables, and text: the summary is only used for
  # the embedding and ANN search)
  def create_ai_summary(self, content: Content) -> str:
    try:
        # Extract content parts
        text = content['text']
        tables = content['tables']
        images = content['images']
        
        # Initialize LLM (needs vision model for images)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # Build tables into string
        tables_text = ""
        if tables:
          for i, table in enumerate(tables):
            tables_text += f"Table {i+1}:\n{table}\n\n"
        
        # Build the text prompt
        prompt_text = f"""You are creating a searchable description for document content retrieval.

        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}

        TABLES:
        {tables_text}

        IMAGES:
        Images will be sent separately through your vision model.
        Also consider images when generating your description of all content provided.

        YOUR TASK:
        Generate a comprehensive, searchable description that covers:

        1. Key facts, numbers, and data points from text and tables
        2. Main topics and concepts discussed  
        3. Questions this content could answer
        4. Visual content analysis (charts, diagrams, patterns in images)
        5. Alternative search terms users might use

        Make it detailed and searchable - prioritize findability over brevity.

        SEARCHABLE DESCRIPTION:

        """

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add images to the message
        for image_base64 in images:
            message_content.append({
          "type": "image_url",
          "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
          print(f"    AI summary failed. Returning text only: {e}")
          return content['text']


  def chunk(self, elements: List[Element]) -> List[Document]:
    elementChunks = chunk_by_title(elements, max_characters=2000, new_after_n_chars=1600, combine_text_under_n_chars=500)
    print("    Combined elements into " + str(len(elementChunks)) + " elementChunks")

    documents = []

    for i, elementChunk in enumerate(elementChunks):
      print(f"    processing chunk {i}")
      content = self.separate_content_types(elementChunk)
      text = content['text']

      # Summarize chunks with tables or images
      if content['tables'] or content['images']:
        print("    found tables or images")
        text = self.create_ai_summary(content)
        print("    generated AI summary")

      # Create langchain document
      documents.append(Document(page_content=text, metadata={"original_content": json.dumps(
                  {"raw_text": content['text'],
                    "tables_html": content['tables'],
                    "images_base64": content['images']})}))
      
    return documents

  def load_and_chunk(self, path: str) -> List[Document]:
    all_chunks: List[Document] = []

    # Get all valid files from the path
    print("DISCOVERING FILES")
    p = Path(path)
    files: List[Path] = []
    for entry in p.iterdir():
      if entry.is_file() and entry.suffix.lower() != ".pkl":
        if entry.suffix.lower() in self.supported_extensions:
          files.append(entry)
        else:
          print("  skipped file " + entry.name + " because its filetype is not supported by MultiModalLoaderChunker")
    
    # Chunk all valid files
    print("CHUNKING FILES")
    for file in files:
      # Create cache file path
      cache_file = file.parent / f"{file.stem}_elements.pkl"
      
      # Check if cached elements exist
      if cache_file.exists():
        print(f"  Loading cached elements for {file.name}")
        with open(cache_file, 'rb') as f:
          elements = pickle.load(f)
      else:
        print(f"  Processing {file.name} elements (first time)")
        elements = self.load(file)
        
        # Save elements to cache
        with open(cache_file, 'wb') as f:
          pickle.dump(elements, f)
          
      print(f"  Chunking {file.name}")
      chunks = self.chunk(elements)
      all_chunks += chunks

    # Return all chunks
    return all_chunks