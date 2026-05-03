"""
Indexing entry point.

Run this once (and again whenever your PDF corpus changes) to:

    1. Ensure required local directories exist
    2. Convert all PDFs in ``DOCS_DIR`` to Markdown
    3. Reset the Qdrant child-chunk collection
    4. Build parent / child chunks and persist them
       (children -> Qdrant, parents -> JSON files)

Usage:
    python ingest.py
"""

import logging

from config import CHILD_COLLECTION, ensure_directories
from embeddings import (
    build_dense_embeddings,
    build_sparse_embeddings,
    get_embedding_dimension,
)
from indexing import index_documents
from text_processing import pdfs_to_markdowns
from vector_store import (
    build_child_vector_store,
    build_client,
    reset_collection,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the full ingestion pipeline."""
    ensure_directories()

    # 1. Convert PDFs to Markdown (skips already-converted files).
    logger.info("Step 1/4: PDF -> Markdown")
    pdfs_to_markdowns()

    # 2. Build embedding models. Dense dim is needed to size the collection.
    logger.info("Step 2/4: Building embedding models")
    dense = build_dense_embeddings()
    sparse = build_sparse_embeddings()
    dim = get_embedding_dimension(dense)

    # 3. Reset the child collection and obtain a vector store handle.
    logger.info("Step 3/4: Resetting Qdrant collection '%s'", CHILD_COLLECTION)
    client = build_client()
    reset_collection(client, CHILD_COLLECTION, dim)
    child_store = build_child_vector_store(client, dense, sparse)

    # 4. Chunk every Markdown file and write to vector + parent stores.
    logger.info("Step 4/4: Indexing chunks")
    index_documents(child_store)

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
