"""
Configuration module for the Agentic RAG application.

Centralizes all paths, collection names, model identifiers, and runtime
parameters so other modules can be swapped without touching configuration.

Environment variables (all optional, sensible defaults applied):
    DOCS_DIR                - Directory containing source PDF files
    MARKDOWN_DIR            - Directory for converted Markdown files
    PARENT_STORE_PATH       - Directory for parent chunk JSON files
    QDRANT_PATH             - Local Qdrant on-disk database path
    CHILD_COLLECTION        - Qdrant collection name for child chunks
    OLLAMA_MODEL            - Ollama chat model identifier
    DENSE_EMBEDDING_MODEL   - HuggingFace sentence-transformers model name
    SPARSE_EMBEDDING_MODEL  - FastEmbed sparse model name
    MIN_PARENT_SIZE         - Minimum parent chunk character length
    MAX_PARENT_SIZE         - Maximum parent chunk character length
    CHILD_CHUNK_SIZE        - Child chunk character length
    CHILD_CHUNK_OVERLAP     - Overlap between child chunks (characters)
    SEARCH_SCORE_THRESHOLD  - Similarity score threshold for child retrieval
    GRADIO_SERVER_NAME      - Host interface for Gradio (default: 127.0.0.1)
    GRADIO_SERVER_PORT      - Port for Gradio (default: 7860)
"""

import os
from pathlib import Path

# Local data layout
DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
MARKDOWN_DIR = Path(os.getenv("MARKDOWN_DIR", "markdown"))
PARENT_STORE_PATH = Path(os.getenv("PARENT_STORE_PATH", "parent_store"))
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", "qdrant_db"))

# Vector store collections
CHILD_COLLECTION = os.getenv("CHILD_COLLECTION", "document_child_chunks")

# Models
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
DENSE_EMBEDDING_MODEL = os.getenv(
    "DENSE_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
)
SPARSE_EMBEDDING_MODEL = os.getenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")

# Chunking parameters
MIN_PARENT_SIZE = int(os.getenv("MIN_PARENT_SIZE", "2000"))
MAX_PARENT_SIZE = int(os.getenv("MAX_PARENT_SIZE", "10000"))
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "500"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "100"))

# Retrieval parameters
SEARCH_SCORE_THRESHOLD = float(os.getenv("SEARCH_SCORE_THRESHOLD", "0.7"))

# Gradio server
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

# Disable HuggingFace tokenizers parallelism warning during multi-process work
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def ensure_directories() -> None:
    """Create all local data directories if they don't already exist."""
    for d in (DOCS_DIR, MARKDOWN_DIR, PARENT_STORE_PATH):
        d.mkdir(parents=True, exist_ok=True)
