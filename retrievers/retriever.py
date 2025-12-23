from abc import ABC, abstractmethod
from typing import List
from rag_types.vector import SemanticCandidate

def rrf(subresults: List[List[SemanticCandidate]], finalK: int, c: int = 60) -> List[SemanticCandidate]:
  if len(subresults) <= 1:
    raise RuntimeError("We need at least 2 subresults to perform RRF")
  
  scores = {}

  # Iterate through every candidate in every subresult
  for subresult in subresults:
    for rank, candidate in enumerate(subresult):
      # Compute RRF score addition
      additional_score = 1 / (c + rank + 1)

      if candidate['id'] not in scores:
        scores[candidate['id']] = [additional_score, candidate]
      else:
        scores[candidate['id']][0] += additional_score
  
  # Take the candidates with the finalK highest RRF scores
  scored = [(tup[0], tup[1]) for tup in scores.values()]
  scored.sort(key = lambda x: x[0], reverse = True)
  return [candidate for _, candidate in scored[:finalK]]

class Retriever(ABC):
  @abstractmethod
  def retrieve_candidates(self, queries: List[str]) -> List[SemanticCandidate]:
    pass