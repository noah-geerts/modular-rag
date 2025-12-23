from typing import List
from rag_types.vector import SemanticCandidate
from retrievers.retriever import Retriever, rrf
from vector_stores.vector_store import VectorStore
from embedders.embedder import Embedder
from concurrent.futures import ThreadPoolExecutor
import itertools

class SemanticRetriever(Retriever):
  def __init__(self, vectorDb: VectorStore, embedder: Embedder, semanticK: int = 10, finalK: int = 3):
    self.vectorDb = vectorDb
    self.embedder = embedder
    self.perQueryK = semanticK
    self.finalK = finalK

  def retrieve_candidates(self, queries: List[str]) -> List[SemanticCandidate]:
    # Check for no queries (makes no sense.. we can't retrieve for nothing)
    N = len(queries)
    if N == 0:
      raise RuntimeError("No queries were provided to the SemanticRetriever's retrieve_candidates method")

    # Embed each query for vector search
    queryVectors = self.embedder.embed_strings(queries)

    # For each query, do retrieval (in parallel to reduce number of network RTT's)
    with ThreadPoolExecutor(max_workers=N) as ex:
      subresults = list(ex.map(self.vectorDb.semantic_search, queryVectors, itertools.repeat(self.perQueryK)))

    # Perform RRF if there is more than one subresult
    if len(subresults) == 1:
      return subresults[0][:self.finalK]
    else:
      return rrf(subresults, self.finalK)