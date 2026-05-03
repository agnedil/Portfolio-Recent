"""
Agent tools for retrieval.

Exposes two LangChain ``@tool`` callables that the agent can invoke during a
ReAct-style loop:

    - ``search_child_chunks``: hybrid semantic + lexical search over child chunks
    - ``retrieve_parent_chunks``: load full parent context for a given ``parent_id``

Two-stage retrieval pattern:
    1. Agent searches child chunks (fast, semantic + BM25).
    2. Agent retrieves the parent of the most relevant child(ren) for full context.

The vector store handle is injected via ``build_tools`` so this module remains
plug-and-play: the tool surface stays stable even if the underlying retrieval
backend changes.

Documentation:
    - https://docs.langchain.com/oss/python/langchain/tools
"""

import json
from pathlib import Path
from typing import Callable, Tuple

from langchain_core.tools import tool

from config import PARENT_STORE_PATH, SEARCH_SCORE_THRESHOLD


def build_tools(child_vector_store) -> Tuple[Callable, Callable]:
    """Build retrieval tools bound to a specific vector store instance.

    Args:
        child_vector_store: A LangChain ``VectorStore`` for child chunks.

    Returns:
        ``(search_child_chunks, retrieve_parent_chunks)`` callables suitable
        for ``llm.bind_tools(...)``.
    """

    @tool
    def search_child_chunks(query: str, limit: int) -> str:
        """Search for the top K most relevant child chunks.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.
        """
        try:
            results = child_vector_store.similarity_search(
                query, k=limit, score_threshold=SEARCH_SCORE_THRESHOLD
            )
            if not results:
                return "NO_RELEVANT_CHUNKS"

            # Compact, agent-friendly text format with parent_id for follow-up calls.
            return "\n\n".join(
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            )
        except Exception as e:
            return f"RETRIEVAL_ERROR: {e}"

    @tool
    def retrieve_parent_chunks(parent_id: str) -> str:
        """Retrieve a full parent chunk by its ID.

        Args:
            parent_id: Parent chunk ID to retrieve.
        """
        # Accept either the bare id or a filename ending in ``.json``.
        file_name = parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
        path = Path(PARENT_STORE_PATH) / file_name

        if not path.exists():
            return "NO_PARENT_DOCUMENT"

        data = json.loads(path.read_text(encoding="utf-8"))
        return (
            f"Parent ID: {parent_id}\n"
            f"File Name: {data.get('metadata', {}).get('source', 'unknown')}\n"
            f"Content: {data.get('page_content', '').strip()}"
        )

    return search_child_chunks, retrieve_parent_chunks
