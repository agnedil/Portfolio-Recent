"""
RAG with LangChain (Hybrid Retrieval + Reranking)
=================================================

A condensed Retrieval-Augmented Generation pipeline that combines lexical
(BM25) and dense (FAISS over Cohere embeddings) retrieval into a weighted
ensemble, applies a Cohere reranker on top, and answers conversationally
with a Replicate-hosted LLM.

This script is a condensed version of
``https://github.com/agnedil/advanced-rag-streamlit-app/blob/main/advanced_rag_history.py``.

When chains are fine vs. when to graduate
-----------------------------------------
- For simple linear RAG (retrieve -> generate -> done), LangChain chains
  work perfectly well.
- LangGraph gives you room to grow without refactoring: extra nodes,
  conditional routing, cycles (self-critique, query rewriting, etc.).

Architecture
------------
1. Load each PDF with ``OnlinePDFLoader`` (all pages, not just the first).
2. Recursive character split into ~1500-character chunks.
3. Two retrievers:
       - BM25Retriever (lexical)
       - FAISS over Cohere dense embeddings
4. Hybrid fusion via ``EnsembleRetriever`` with weights [0.6, 0.4].
5. ``CohereRerank`` compresses the fused candidates to top_n results.
6. ``ConversationalRetrievalChain`` answers with the Replicate LLM and
   returns the source documents alongside the answer.

Setup
-----
1. Install dependencies::

       pip install langchain langchain-community langchain-cohere \\
           langchain-text-splitters cohere replicate faiss-cpu \\
           unstructured[pdf] rank_bm25 pypdf

2. Set API keys before running::

       export REPLICATE_API_TOKEN=...
       export COHERE_API_KEY=...

3. Fill in the PDF URLs in ``PDF_URLS`` at the bottom of this file, then::

       python rag_advanced_langchain_faiss_llama.py

Notes
-----
- The script refuses to start if either API key is missing.
- ``OnlinePDFLoader`` requires ``unstructured[pdf]`` (and its ``poppler``
  system dependency) to parse PDFs.
- The Replicate model id may need updating as Meta releases newer Llama
  versions; see ``LLM_MODEL`` below.
"""

import os
import sys
from typing import List, Optional, Tuple

# Must be set before HuggingFace tokenizers initialize, hence early.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.llms import Replicate
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------- Configuration ------------------------------ #

# Fill in the URLs of the PDFs you want to index.
PDF_URLS: List[str] = [
    "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf",
    "https://arxiv.org/pdf/2402.05120",
]

# Chunking and retrieval parameters.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100
TOP_K = 5

# Hybrid fusion weights for [BM25, FAISS].
ENSEMBLE_WEIGHTS = [0.6, 0.4]

# Cohere embedding model used by FAISS.
EMBED_MODEL = "embed-english-light-v3.0"

# Replicate-hosted chat LLM. Update if Replicate retires this model.
LLM_MODEL = "meta/meta-llama-3-70b"
LLM_KWARGS = {"temperature": 0.5, "top_p": 1, "max_new_tokens": 1000}

# Required environment variables (checked at startup).
REQUIRED_ENV = ("REPLICATE_API_TOKEN", "COHERE_API_KEY")


def _require_env(*names: str) -> None:
    """Exit with a clear message if any of the named env vars is unset."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        # Print to stderr and exit so the user sees a clear, actionable error.
        sys.stderr.write(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + "\nSet them in your shell, e.g.\n"
            "    export REPLICATE_API_TOKEN=...\n"
            "    export COHERE_API_KEY=...\n"
        )
        sys.exit(1)


def create_rag_chain(
    pdf_links: List[str],
    chunk_size: int = CHUNK_SIZE,
    top_k: int = TOP_K,
) -> ConversationalRetrievalChain:
    """Build an advanced RAG chain with hybrid retrieval and reranking.
    Args:
        pdf_links: List of HTTPS URLs pointing at PDF files.
        chunk_size: Target chunk size in characters.
        top_k: Number of candidates each retriever returns and the
            reranker keeps.
    Returns:
        A ``ConversationalRetrievalChain`` ready to be invoked.
    Raises:
        ValueError: If ``pdf_links`` is empty or contains blank entries.
    """
    # Validate inputs early; a blank URL produces a confusing failure deep
    # inside OnlinePDFLoader.
    if not pdf_links or not all(url.strip() for url in pdf_links):
        raise ValueError(
            "pdf_links must be a non-empty list of HTTPS URLs. "
            "Edit PDF_URLS at the top of this file."
        )

    # 1. Load every page of each PDF (the original used [0], dropping pages).
    docs = []
    for url in pdf_links:
        docs.extend(OnlinePDFLoader(url).load())

    # 2. Recursive split into retrieval-friendly chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    # 3a. Lexical retriever: BM25 over the same chunks.
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = top_k

    # 3b. Dense retriever: FAISS over Cohere embeddings.
    faiss_store = FAISS.from_documents(chunks, CohereEmbeddings(model=EMBED_MODEL))
    faiss = faiss_store.as_retriever(search_kwargs={"k": top_k})

    # 4. Hybrid fusion: weighted ensemble of lexical + dense candidates.
    ensemble = EnsembleRetriever(retrievers=[bm25, faiss], weights=ENSEMBLE_WEIGHTS)

    # 5. Cohere reranker compresses the fused candidates to the top_k.
    retriever = ContextualCompressionRetriever(
        base_retriever=ensemble,
        base_compressor=CohereRerank(top_n=top_k),
    )

    # 6. LLM + conversational chain that returns retrieved sources too.
    llm = Replicate(model=LLM_MODEL, model_kwargs=LLM_KWARGS)
    return ConversationalRetrievalChain.from_llm(
        llm, retriever, return_source_documents=True
    )


def query_with_history(
    chain: ConversationalRetrievalChain,
    query: str,
    history: Optional[List[Tuple[str, str]]] = None,
    max_history: int = 5,
) -> Tuple[str, List[Tuple[str, str]]]:
    """Query the chain and keep the last few turns as chat history.
    Args:
        chain: A chain returned by ``create_rag_chain``.
        query: The user's question.
        history: Prior ``(question, answer)`` pairs. ``None`` starts fresh.
        max_history: Maximum number of recent turns to retain.
    Returns:
        ``(answer, history)`` where ``history`` is trimmed to the most
        recent ``max_history`` turns.
    """
    history = list(history) if history else []
    result = chain.invoke({"question": query, "chat_history": history})
    answer = result["answer"]
    history.append((query, answer))
    return answer, history[-max_history:]


# ----------------------------- Main --------------------------------------- #

if __name__ == "__main__":
    # Fail fast if the required API keys are missing.
    _require_env(*REQUIRED_ENV)

    # Build the chain once and reuse it across turns.
    chain = create_rag_chain(PDF_URLS)
    history: List[Tuple[str, str]] = []

    print("RAG chain ready. Type a question (empty line to quit).")
    while True:
        # Graceful exit on EOF / Ctrl-C as well as empty input.
        try:
            q = input("\nAsk: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break

        answer, history = query_with_history(chain, q, history)
        print("\nAnswer:\n", answer)
