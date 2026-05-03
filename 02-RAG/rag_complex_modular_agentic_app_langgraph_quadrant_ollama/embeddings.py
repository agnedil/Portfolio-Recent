"""
Embeddings module.

Configures embedding models for hybrid retrieval (dense + sparse):
    - Dense embeddings capture semantic meaning via a transformer model
    - Sparse embeddings (BM25) provide lexical / keyword matching

Replace either factory to use a different embedding strategy without touching
the vector store, indexing, or retrieval code.

Documentation:
    - https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub
    - https://qdrant.github.io/fastembed/
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse

from config import DENSE_EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL


def build_dense_embeddings() -> HuggingFaceEmbeddings:
    """Build dense embeddings using a HuggingFace sentence-transformers model."""
    return HuggingFaceEmbeddings(model_name=DENSE_EMBEDDING_MODEL)


def build_sparse_embeddings() -> FastEmbedSparse:
    """Build sparse BM25 embeddings via FastEmbed."""
    return FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL)


def get_embedding_dimension(embeddings: HuggingFaceEmbeddings) -> int:
    """Return the output dimension of a dense embedding model.

    Used at collection-creation time to size the dense vector field.
    """
    return len(embeddings.embed_query("dimension probe"))
