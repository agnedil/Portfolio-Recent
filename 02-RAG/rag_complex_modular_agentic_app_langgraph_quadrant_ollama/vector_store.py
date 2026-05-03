"""
Vector store module.

Wraps the Qdrant client and collection lifecycle and exposes a hybrid
(dense + sparse) vector store handle suitable for both indexing and retrieval.

Replace this module to plug in a different vector database (Pinecone,
Weaviate, Milvus, pgvector, etc.). Downstream code only depends on the
LangChain ``VectorStore`` interface, so the rest of the application keeps
working unchanged.

Documentation:
    - https://python.langchain.com/docs/integrations/vectorstores/qdrant
"""

import logging

from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import CHILD_COLLECTION, QDRANT_PATH

logger = logging.getLogger(__name__)


def build_client() -> QdrantClient:
    """Build a local file-backed Qdrant client."""
    return QdrantClient(path=str(QDRANT_PATH))


def ensure_collection(
    client: QdrantClient, collection_name: str, embedding_dimension: int
) -> None:
    """Create a hybrid (dense + sparse) collection if it does not already exist."""
    if client.collection_exists(collection_name):
        logger.info("Collection already exists: %s", collection_name)
        return

    # Dense vectors with cosine distance, plus a named sparse field for BM25.
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=embedding_dimension,
            distance=qmodels.Distance.COSINE,
        ),
        sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
    )
    logger.info("Created collection: %s", collection_name)


def reset_collection(
    client: QdrantClient, collection_name: str, embedding_dimension: int
) -> None:
    """Drop and recreate a collection. Used during full re-indexing."""
    if client.collection_exists(collection_name):
        logger.info("Removing existing Qdrant collection: %s", collection_name)
        client.delete_collection(collection_name)
    ensure_collection(client, collection_name, embedding_dimension)


def build_child_vector_store(
    client: QdrantClient,
    dense_embeddings,
    sparse_embeddings,
    collection_name: str = CHILD_COLLECTION,
) -> QdrantVectorStore:
    """Return the hybrid retrieval vector store for child chunks."""
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="sparse",
    )
