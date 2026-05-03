"""
Build the Chroma vector store from text documents.

Loads every regular file under ``DOCS_DIR``, splits each document with a
recursive character splitter, embeds the chunks with OpenAI embeddings, and
persists them to a local Chroma database whose location and collection name
are imported from ``build_rag_graph`` to keep the indexer and the graph in
sync.

Run this once before serving the app, and again whenever your document
corpus changes.

Usage:
    export OPENAI_API_KEY=...
    python index.py
"""

import logging
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from build_rag_graph import COLLECTION_NAME, EMBEDDING_MODEL, PERSIST_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Source documents and chunking parameters.
DOCS_DIR = Path("docs")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def load_docs(docs_dir: Path = DOCS_DIR) -> List[Document]:
    """Load every regular file under ``docs_dir`` as LangChain Documents.

    Args:
        docs_dir: Directory containing the source text files.

    Returns:
        A flat list of ``Document`` objects, one per file.

    Raises:
        FileNotFoundError: If the directory is missing or contains no files.
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"Document directory not found: {docs_dir}/")

    # Use TextLoader on every regular file; works for plain .txt / .md / etc.
    docs = [
        doc
        for fp in docs_dir.glob("*")
        if fp.is_file()
        for doc in TextLoader(str(fp), encoding="utf-8").load()
    ]
    if not docs:
        raise FileNotFoundError(f"No documents found in {docs_dir}/")
    return docs


def build_index() -> None:
    """Load, chunk, embed, and persist documents to Chroma."""
    # 1. Load raw text files.
    docs = load_docs()
    logger.info("Loaded %d documents from %s/", len(docs), DOCS_DIR)

    # 2. Recursive split for retrieval-friendly chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks", len(chunks))

    # 3. Embed and persist (modern Chroma persists automatically).
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    logger.info("Index built at %s/ (collection=%s)", PERSIST_DIR, COLLECTION_NAME)


if __name__ == "__main__":
    build_index()
