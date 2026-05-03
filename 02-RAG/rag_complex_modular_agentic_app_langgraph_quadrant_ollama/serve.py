"""
Application entry point.

Loads the existing vector store (built by ``ingest.py``), wires up the
LangGraph agent, and launches the Gradio web app.

Usage:
    python serve.py
"""

import logging

from app import launch
from config import ensure_directories
from embeddings import build_dense_embeddings, build_sparse_embeddings
from graph import build_agent_graph
from llm import build_llm
from nodes import build_nodes
from tools import build_tools
from vector_store import build_child_vector_store, build_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Wire dependencies and launch the Gradio interface."""
    ensure_directories()

    # 1. Storage / retrieval layer.
    logger.info("Connecting to vector store...")
    client = build_client()
    dense = build_dense_embeddings()
    sparse = build_sparse_embeddings()
    child_store = build_child_vector_store(client, dense, sparse)

    # 2. Agent tools and LLM (with tools bound for the inner ReAct loop).
    logger.info("Building agent tools and LLM...")
    search_tool, retrieve_tool = build_tools(child_store)
    llm = build_llm(temperature=0.0)
    llm_with_tools = llm.bind_tools([search_tool, retrieve_tool])

    # 3. Compile the LangGraph orchestrator.
    logger.info("Compiling agent graph...")
    nodes = build_nodes(llm=llm, llm_with_tools=llm_with_tools)
    agent_graph = build_agent_graph(nodes, tools=[search_tool, retrieve_tool])

    # 4. Serve the UI.
    launch(agent_graph)


if __name__ == "__main__":
    main()
