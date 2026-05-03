"""
RAG LangGraph definition.

Builds a minimal two-node graph:

    START -> retrieve -> generate -> END

where ``retrieve`` fetches the top-k chunks from the persisted Chroma
collection and ``generate`` answers strictly from the retrieved context using
a chat LLM.

This module also owns the storage / model constants (``PERSIST_DIR``,
``COLLECTION_NAME``, ``EMBEDDING_MODEL``) so ``index.py`` can import them and
both sides of the pipeline always agree.

The graph is constructed inside ``build_graph()`` rather than at import time,
so importing this module does not touch the vector store or contact OpenAI.
"""

from typing import Callable, List, TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph

# Storage and model configuration. index.py imports these to stay in sync.
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 4

# Grounded-answer instruction sent to the LLM at every generation step.
SYSTEM_PROMPT = (
    "Answer using ONLY the context. "
    "If the answer is not in the context, say you don't have that "
    "information in the documents."
)


class RAGState(TypedDict):
    """Explicit state object passed between graph nodes."""
    question: str
    docs: List[Document]
    answer: str


def _build_retriever():
    """Open the persisted Chroma collection and return a retriever."""
    # Re-uses the on-disk DB created by index.py
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


def _build_llm() -> ChatOpenAI:
    """Build the chat LLM used for grounded generation."""
    return ChatOpenAI(model=CHAT_MODEL, temperature=0)


def _make_retrieve_node(retriever) -> Callable[[RAGState], RAGState]:
    """Return the ``retrieve`` node bound to a specific retriever."""

    def retrieve(state: RAGState) -> RAGState:
        # Look up the top-k most similar chunks for the question.
        q = state["question"]
        docs = retriever.invoke(q)
        return {"question": q, "docs": docs, "answer": ""}

    return retrieve


def _make_generate_node(llm: ChatOpenAI) -> Callable[[RAGState], RAGState]:
    """Return the ``generate`` node bound to a specific chat LLM."""

    def generate(state: RAGState) -> RAGState:
        q, docs = state["question"], state["docs"]
        # Concatenate retrieved chunk text into a single context block.
        context = "\n\n".join(d.page_content for d in docs)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Question:\n{q}\n\nContext:\n{context}"),
        ]
        resp = llm.invoke(messages)
        return {**state, "answer": resp.content}

    return generate


def build_graph():
    """Compile and return the RAG graph.

    Easy to extend: insert nodes (e.g. ``rewrite_query``, ``rerank``,
    ``evaluate_response``) and rewire edges without touching the rest.

    Returns:
        A compiled LangGraph runnable accepting a ``RAGState`` dict.
    """
    # Lazy: build store and LLM only when the graph is actually compiled.
    retriever = _build_retriever()
    llm = _build_llm()

    # Wire the two-node pipeline.
    g = StateGraph(RAGState)
    g.add_node("retrieve", _make_retrieve_node(retriever))
    g.add_node("generate", _make_generate_node(llm))
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()
