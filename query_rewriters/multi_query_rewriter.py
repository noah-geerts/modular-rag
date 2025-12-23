from query_rewriters.query_rewriter import QueryRewriter
from typing import List
from openai import OpenAI

class MultiQueryRewriter(QueryRewriter):
  def __init__(self, openai_api_key, n: int = 3):
    self.client = OpenAI(api_key=openai_api_key)
    self.n = n

  # Writes multiple variations of a query with better wording
  def rewrite_query(self, query: str) -> List[str]:
    # System prompt explaining what the LLM is
    system_prompt = """You are a retrieval assistant for a Retrieval-Augmented Generation (RAG) system.
    Your task is to generate multiple alternative search queries that preserve the original intent while improving retrieval coverage.
    """

    # User prompt asking it to perform multi-query
    user_prompt = f"""Given the following user query, generate {self.n} alternative search queries.

    Goals:
    - Preserve the original intent exactly (no new constraints or assumptions).
    - Improve clarity by making implicit details explicit when appropriate.
    - Include more technical or domain-specific terminology where helpful.
    - Use varied phrasing and synonyms to increase retrieval recall.

    Guidelines:
    - Do NOT answer the query.
    - Do NOT explain your reasoning.
    - Each query must be a standalone search query.
    - Keep queries concise and retriever-friendly.
    - Do NOT number or bullet the queries.
    - Separate each query using the exact delimiter: |--|

    Original query:
    {query}

    Output:
    Return only the queries, separated by |--|, and nothing else.
    """

    # Send to AI and get response
    response = self.client.chat.completions.create(
        model="gpt-4.1",
        messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
        ]
    )

    # Split and return
    if response.choices[0].message.content is None:
      raise RuntimeError("response.choices[0].message.content received from the LLM was None")
    queries = response.choices[0].message.content.split("|--|")
    queries = [query.strip() for query in queries]
    return queries