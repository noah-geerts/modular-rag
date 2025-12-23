import unittest
from unittest.mock import MagicMock, patch, call

from retrievers.semantic_retriever import SemanticRetriever


class TestSemanticRetriever(unittest.TestCase):
  def test_no_queries_throws_error(self):
    # Arrange (embedder returns one vector, semantic search returns 3 candidates)
    embedder = MagicMock()
    vector_db = MagicMock()
    embedder.embed_strings.return_value = []
    vector_db.semantic_search.return_value = []

    # Act (call with ONE query)
    retriever = SemanticRetriever(vector_db, embedder, semanticK=3, finalK=2)

    with patch('retrievers.semantic_retriever.rrf') as mock_rrf:
      self.assertRaises(RuntimeError, lambda: retriever.retrieve_candidates([]))

    # Assert: none of the subfunctions should have been called
    embedder.embed_strings.assert_not_called()
    vector_db.semantic_search.assert_not_called()
    mock_rrf.assert_not_called()

  def test_single_query_returns_top_k_without_rrf(self):
    # Arrange (embedder returns one vector, semantic search returns 3 candidates)
    embedder = MagicMock()
    vector_db = MagicMock()
    embedder.embed_strings.return_value = ["vec1"]
    vector_db.semantic_search.return_value = [
      {"id": 1, "score": 1.2},
      {"id": 2, "score": 1.2},
      {"id": 3, "score": 1.2},
    ]

    # Act (call with ONE query)
    retriever = SemanticRetriever(vector_db, embedder, semanticK=3, finalK=2)

    with patch('retrievers.semantic_retriever.rrf') as mock_rrf:
      result = retriever.retrieve_candidates(["q1"])

    # Assert
    mock_rrf.assert_not_called() # should get no RRF with one query
    embedder.embed_strings.assert_called_once_with(["q1"]) # should embed the query
    vector_db.semantic_search.assert_called_once_with("vec1", 3) # should perform one semantic
    # search for the one query with k = 3
    self.assertEqual(result, [{"id": 1, "score": 1.2}, {"id": 2, "score": 1.2},]) # should simply
    # truncate to the top finalK = 2 because we only have one subquery

  def test_multiple_queries_uses_rrf_and_combines_subresults(self):
    # Arrange: this time we have two embeddings for 2 queries, and we will have 2 semantic search outputs
    embedder = MagicMock()
    vector_db = MagicMock()
    embedder.embed_strings.return_value = ["v1", "v2"]
    result_q1 = [{"id": 1, "score": 1.2}, {"id": 2, "score": 1.2},]
    result_q2 = [{"id": 3, "score": 1.2}, {"id": 4, "score": 1.2}]
    vector_db.semantic_search.side_effect = [result_q1, result_q2] # first call returns result_q1, second result_q2

    # Act
    retriever = SemanticRetriever(vector_db, embedder, semanticK=2, finalK=1)
    rrf_result = [{"id": 1, "score": 1.1}]

    with patch('retrievers.semantic_retriever.rrf', return_value=rrf_result) as mock_rrf:
      result = retriever.retrieve_candidates(["q1", "q2"]) # 2 queries

    # Assert: Ensure we get 2 queries embedding, and the 2 embeddings passed into semantic search
    embedder.embed_strings.assert_called_once_with(["q1", "q2"])
    vector_db.semantic_search.assert_has_calls([call("v1", 2), call("v2", 2)])
    mock_rrf.assert_called_once_with([result_q1, result_q2], 1) # And ensure rrf is called
    self.assertEqual(result, rrf_result)

  def test_multiple_queries_still_calls_rrf_when_some_results_empty(self):
    # Arrange: embedder returns 3 vectors for 3 queries, semantic search returns empty for q1 and q3, non-empty for q2
    embedder = MagicMock()
    vector_db = MagicMock()
    embedder.embed_strings.return_value = ["v1", "v2", "v3"]
    result_q1 = []
    result_q2 = [{"id": 1, "score": 1.1}]
    result_q3 = []
    vector_db.semantic_search.side_effect = [result_q1, result_q2, result_q3]

    # Act: semanticK=1, finalK=2, call with 3 queries
    retriever = SemanticRetriever(vector_db, embedder, semanticK=1, finalK=2)
    rrf_result = [{"id": 1, "score": 1.1}]

    with patch('retrievers.semantic_retriever.rrf', return_value=rrf_result) as mock_rrf:
      result = retriever.retrieve_candidates(["q1", "q2", "q3"])

    # Assert: rrf should be called with all results (including empty), and finalK=2
    mock_rrf.assert_called_once_with([result_q1, result_q2, result_q3], 2)
    self.assertEqual(result, rrf_result)


if __name__ == '__main__':
  unittest.main()
