from typing import List
import unittest

from rag_types.vector import SemanticCandidate
from retrievers.retriever import rrf

class TestRRF(unittest.TestCase):
  def test_rrf_ranks_in_expected_order(self):
    # Arrange (2 score first twice and is in 3, 1 scores first once and is in 3, and then 3 is in 2)
    subresults: List[List[SemanticCandidate]] = [
      [{'id': 1, 'score': 2.2}, {'id': 2, 'score': 2.2}, {'id': 3, 'score': 2.2}],
      [{'id': 2, 'score': 2.2}, {'id': 3, 'score': 2.2}, {'id': 1, 'score': 2.2}],
      [{'id': 2, 'score': 2.2}, {'id': 4, 'score': 2.2}, {'id': 1, 'score': 2.2}],
    ]

    # Act & Assert
    ranked = rrf(subresults, finalK = 3)
    self.assertEqual([candidate['id'] for candidate in ranked], [2, 1, 3])

  def test_rrf_returns_nothing_with_empty_subresults(self):
    # Arrange
    subresults = [[], [], []]

    # Act & Assert
    ranked = rrf(subresults, finalK = 3)
    self.assertEqual(ranked, [])

  def test_rrf_requires_multiple_subresults(self):
    # Should throw an error with only one subarray
    with self.assertRaises(RuntimeError):
      rrf([[]], finalK = 2)

    # Should throw an error with NO subarrays
    with self.assertRaises(RuntimeError):
      rrf([], finalK = 2)

  def test_rrf_handles_empty_lists_and_truncates(self):
    # Arrange: When one list is empty, we should get all results from the other,
    # truncating if we don't hit finalK outputs
    subresults: List[List[SemanticCandidate]] = [
      [],
      [{'id': 1, 'score': 1.0}, {'id': 2, 'score': 1.0}],
    ]

    # Act & Assert
    ranked = rrf(subresults, finalK = 5)
    self.assertEqual([candidate['id'] for candidate in ranked], [1, 2])

  def test_rrf_respects_finalk_and_aggregates_scores(self):
    # Arrange
    subresults: List[List[SemanticCandidate]] = [
      [{'id': 2, 'score': 2.2}, {'id': 1, 'score': 2.2}],
      [{'id': 1, 'score': 2.2}, {'id': 3, 'score': 2.2}],
    ]

    # Act & Assert (ensure we only 1 and 2 since finalK is 2)
    ranked = rrf(subresults, finalK = 2)
    self.assertEqual([candidate['id'] for candidate in ranked], [1, 2])
    self.assertEqual(len(ranked), 2)


if __name__ == '__main__':
  unittest.main()
