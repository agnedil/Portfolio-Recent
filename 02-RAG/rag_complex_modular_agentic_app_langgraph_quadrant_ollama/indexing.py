"""
Indexing module.

Orchestrates the per-document pipeline that populates the vector store and the
parent JSON store:

    Markdown file -> parent / child chunks -> child vector store + parent JSON

Parent chunks are persisted as one JSON file per chunk on local disk for fast
random-access retrieval by the agent. Child chunks are written to the
configured hybrid vector store.

Replace this module to change persistence semantics (e.g., parent chunks in
Redis or S3, child chunks in a different vector backend) without touching the
chunking strategy or the agent.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from chunking import build_parent_child_chunks
from config import MARKDOWN_DIR, PARENT_STORE_PATH

logger = logging.getLogger(__name__)


def _read_markdown_files(md_dir: Path) -> List[Path]:
    """Return a sorted list of Markdown files in ``md_dir``."""
    return sorted(md_dir.glob("*.md"))


def _save_parent_chunks(parent_pairs: List[Tuple[str, Document]]) -> None:
    """Persist parent chunks as individual JSON files in ``PARENT_STORE_PATH``."""
    PARENT_STORE_PATH.mkdir(parents=True, exist_ok=True)

    # Wipe previous parent store contents to avoid stale references.
    for item in PARENT_STORE_PATH.iterdir():
        if item.is_file():
            item.unlink()

    for parent_id, doc in parent_pairs:
        payload = {"page_content": doc.page_content, "metadata": doc.metadata}
        path = PARENT_STORE_PATH / f"{parent_id}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def index_documents(child_vector_store: QdrantVectorStore) -> None:
    """Walk ``MARKDOWN_DIR``, chunk every file, and write to vector + parent stores.

    Args:
        child_vector_store: A LangChain ``VectorStore`` configured for hybrid
            retrieval where child chunks will be written.
    """
    md_files = _read_markdown_files(MARKDOWN_DIR)
    if not md_files:
        logger.warning("No .md files found in %s/", MARKDOWN_DIR)
        return

    all_parent_pairs: List[Tuple[str, Document]] = []
    all_children: List[Document] = []

    # 1. Read each Markdown file and produce parent/child chunks.
    for doc_path in md_files:
        logger.info("Processing: %s", doc_path.name)
        try:
            md_text = doc_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("Error reading %s: %s", doc_path.name, e)
            continue

        parent_pairs, children = build_parent_child_chunks(md_text, doc_path.stem)
        all_parent_pairs.extend(parent_pairs)
        all_children.extend(children)

    if not all_children:
        logger.warning("No child chunks to index")
        return

    # 2. Write child chunks to the vector store.
    logger.info("Indexing %d child chunks into vector store...", len(all_children))
    try:
        child_vector_store.add_documents(all_children)
        logger.info("Child chunks indexed successfully")
    except Exception as e:
        logger.error("Error indexing child chunks: %s", e)
        return

    # 3. Persist parent chunks to local JSON.
    logger.info("Saving %d parent chunks to JSON...", len(all_parent_pairs))
    _save_parent_chunks(all_parent_pairs)
    logger.info("Indexing complete.")
