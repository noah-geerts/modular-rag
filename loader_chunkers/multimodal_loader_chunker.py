import json
import pickle
from typing import List, Set
from loader_chunker import LoaderChunker
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from pathlib import Path
from openai import OpenAI
from rag_types.chunk import Chunk, Content

class MultiModalLoaderChunker(LoaderChunker):

  def __init__(self, openai_api_key):
    self.client = OpenAI(api_key=openai_api_key)

  # Generates an AI summary of a chunk containing images and tables that it can be searched by
  # (we will still return the original images, tables, and text: the summary is only used for
  # the embedding and ANN search)
  def generate_ai_summary(self, content: Content) -> str:
    try:
        # Extract content parts
        text = content['text']
        tables = content['tables']
        images = content['images']

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
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "user", "content": message_content}
            ]
        )
        
        return response.content
        
    except Exception as e:
          print(f"    AI summary failed. Returning text only: {e}")
          print(f"    Content that caused error: ")
          return content['text']

  @property
  def supported_extensions(self) -> Set[str]:
    return {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".txt", ".md", ".pdf"}
  
  # Loads files using unstructured, injecting kwargs based on the file type
  def load(self, file: Path) -> List[Element]:
    suffix = file.suffix.lower()

    kwargs = {"filename": str(file)}

    if suffix == ".pdf":
      kwargs |= {
        "strategy": "hi_res",  # High accuracy OCR and layout detection
        "extract_images_in_pdf": True,  # Extract images from PDF
        "infer_table_structure": True,  # Detect and preserve table structure
        "extract_image_block_types": ["Image", "Table"],  # Include images and tables
        "extract_image_block_to_payload": True
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
  def extract_chunk_contents(self, composite_elements: List[Element]) -> List[Content]: 
    chunk_contents = []

    # Separate each element chunk into its content types
    for i, composite_element in enumerate(composite_elements):
      print(f"    Extracting content from elementChunk {i}")
      content: Content = {
        'text': composite_element.text,
        'tables': [],
        'images': []
      }

      if hasattr(composite_element, 'metadata') and hasattr(composite_element.metadata, 'orig_elements'):
        for element in composite_element.metadata.orig_elements:
          element_type = type(element).__name__

          # Tables
          if element_type == 'Table':
            table_html = getattr(element.metadata, 'text_as_html', element.text)
            content['tables'].append(table_html)

          # Images
          elif element_type == 'Image':
            print(f"    -found an image: {composite_element.metadata.orig_elements}")
            print(f"    -image: {type(element)}")

            # Debug: print all available metadata attributes
            if hasattr(element, 'metadata'):
              print(f"    -metadata attributes: {dir(element.metadata)}")
              print(f"    -metadata dict: {vars(element.metadata)}")

            if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
              content['images'].append(element.metadata.image_base64)

        print(f"    content: {content}")
        print()
      chunk_contents.append(content)

    return chunk_contents
    
  def create_chunks(self, chunk_contents: List[Content]) -> List[Chunk]:
    chunks = []

    for i, content in enumerate(chunk_contents):
      print(f"    Creating chunk object for chunk content {i}")
      search_text = content['text']

      # if the chunk contains tables or images, create a searchable summary
      if content['images'] or content['tables']:
        print(f"    -Detected images or tables. Generating AI summary")
        print(f"    -content: {content}")
        search_text = self.generate_ai_summary(content)

      # Construct the lchunk object
      chunk: Chunk = {"search_text": search_text, content: content}
      chunks.append(chunk)

    return chunks

  def load_and_chunk(self, path: str) -> List[Chunk]:
    all_chunks: List[Chunk] = []

    # Get all valid files from the path
    print("DISCOVERING FILES")
    p = Path(path)
    files: List[Path] = []
    for entry in p.iterdir():
      if entry.is_file() and entry.suffix.lower() != ".pkl":
        if entry.suffix.lower() in self.supported_extensions:
          print("  found file: " + entry.name)
          files.append(entry)
        else:
          print("  skipped file " + entry.name + " because its filetype is not supported by MultiModalLoaderChunker")
    
    print("PROCESSING FILES")
    for file in files:

      # Load file with unstructured unless it is already cached
      print(f"  LOADING {file.name}")
      cache_file = file.parent / f"{file.stem}_elements.pkl"
      
      if cache_file.exists():
        print(f"  -Loading cached elements for {file.name}")
        with open(cache_file, 'rb') as f:
          elements = pickle.load(f)
      else:
        print(f"  -Processing {file.name} elements (first time)")
        elements = self.load(file)
        
        with open(cache_file, 'wb') as f:
          pickle.dump(elements, f)
      
      # Use unstructured to create chunks from elements
      print(f"  CHUNKING {file.name}")
      composite_elements = chunk_by_title(elements, max_characters=2000, new_after_n_chars=1600, combine_text_under_n_chars=500)
      print(f"  -Combined {len(elements)} elements into {len(composite_elements)} elementChunks")

      # Extract text, images, and tables from unstructured elementChunks
      print(f"  EXTRACTING tables, images, text from {file.name}'s elementChunks")
      chunk_contents = self.extract_chunk_contents(composite_elements)

      # Convert multimodal chunks to Chunks for storage
      print(f"  CREATING chunk objects for {file.name}")
      chunks = self.create_chunks(chunk_contents)
      all_chunks += chunks

    # Return all chunks
    return all_chunks