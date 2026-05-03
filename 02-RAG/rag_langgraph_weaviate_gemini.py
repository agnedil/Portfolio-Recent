"""
RAG with LangGraph + Weaviate (Gemini Flash)
============================================

A minimal Retrieval-Augmented Generation pipeline that indexes a text
document into an embedded Weaviate vector store and answers questions
through a LangGraph two-node workflow (``retrieve -> generate``) backed
by Google's Gemini Flash chat model and Google Generative AI embeddings.

Architecture
------------
1. Download the source document once and cache it locally.
2. Chunk it with a ``CharacterTextSplitter`` (500 chars, 50-char overlap).
3. Embed and index the chunks into an embedded Weaviate instance.
4. Build a two-node LangGraph: ``retrieve -> generate -> END``.
5. Run sample queries against the compiled graph.

Setup
-----
1. Install dependencies (``weaviate-client`` is pinned to v3 because the
   v3 ``EmbeddedOptions`` API is what
   ``langchain_community.vectorstores.Weaviate`` wraps; v4 uses a different
   constructor and is not compatible with this code path)::

       pip install langchain langchain-community langchain-core \\
           langchain-google-genai langchain-text-splitters langgraph \\
           "weaviate-client<4" requests

2. Set your Google API key in the shell *before* launching::

       export GOOGLE_API_KEY=...

3. Run the app::

       python rag_langgraph_weaviate_gemini.py

Notes
-----
- The script refuses to start if ``GOOGLE_API_KEY`` is missing.
- Embedded Weaviate downloads its server binary on first launch; that
  may take a few seconds.
- The State-of-the-Union text is fetched once and cached locally; delete
  ``state_of_the_union.txt`` to force a re-download.
- The original script downloaded the GitHub HTML page (``/blob/master/...``)
  rather than the raw text; this version uses the ``raw.githubusercontent.com``
  URL so the indexed content is actual prose, not HTML markup.
"""

import os
import sys
from pathlib import Path
from typing import List, TypedDict

import requests
import weaviate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Weaviate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter
from langgraph.graph import END, StateGraph
from weaviate.embedded import EmbeddedOptions

# ----------------------------- Configuration ------------------------------ #

# Raw text URL (the original /blob/ URL returned GitHub's HTML wrapper).
DATA_URL = (
    "https://raw.githubusercontent.com/langchain-ai/langchain/master/"
    "docs/docs/how_to/state_of_the_union.txt"
)
DATA_PATH = Path("state_of_the_union.txt")

# Chunking parameters.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Google Generative AI model identifiers.
CHAT_MODEL = "gemini-2.0-flash"
EMBED_MODEL = "models/text-embedding-004"

# Required environment variable(s) checked at startup.
REQUIRED_ENV = ("GOOGLE_API_KEY",)

# Prompt template used by the generate node.
PROMPT_TEMPLATE = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""


# ----------------------------- Helpers ------------------------------------ #

def _require_env(*names: str) -> None:
    """Exit with a clear message if any of the named env vars is unset."""
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        # Print to stderr and exit so the user sees an actionable error.
        sys.stderr.write(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + "\nSet it in your shell, e.g.\n"
            "    export GOOGLE_API_KEY=...\n"
        )
        sys.exit(1)


# ----------------------------- Data prep ---------------------------------- #

def download_data(url: str = DATA_URL, path: Path = DATA_PATH) -> Path:
    """Download the source text file once and cache it locally.
    Args:
        url: Raw-text URL to fetch.
        path: Local cache path.
    Returns:
        Path to the cached file.
    """
    # Skip re-download if the file is already cached.
    if path.exists():
        return path

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return path


def load_and_chunk_documents(path: Path) -> List[Document]:
    """Load a text file and split it into retrieval-friendly chunks.
    Args:
        path: Local path to a UTF-8 text file.
    Returns:
        A list of LangChain ``Document`` chunks.
    """
    # TextLoader returns one Document per file; the splitter does the rest.
    loader = TextLoader(str(path), encoding="utf-8")
    documents = loader.load()
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks: List[Document]) -> Weaviate:
    """Spin up an embedded Weaviate instance and index the chunks.
    Args:
        chunks: Pre-split LangChain Documents to embed and store.
    Returns:
        A LangChain ``Weaviate`` vector store backed by an embedded server.
    """
    # Embedded Weaviate runs in-process; no external server to manage.
    client = weaviate.Client(embedded_options=EmbeddedOptions())
    return Weaviate.from_documents(
        client=client,
        documents=chunks,
        embedding=GoogleGenerativeAIEmbeddings(model=EMBED_MODEL),
        by_text=False,
    )


# ----------------------------- LangGraph ---------------------------------- #

class RAGGraphState(TypedDict):
    """Explicit state object passed between graph nodes."""
    question: str
    documents: List[Document]
    generation: str


def make_retrieve_node(retriever):
    """Build the ``retrieve`` node bound to a specific retriever.
    Args:
        retriever: A LangChain retriever exposing ``.invoke(query)``.
    Returns:
        A node callable suitable for ``StateGraph.add_node``.
    """

    def retrieve_documents_node(state: RAGGraphState) -> RAGGraphState:
        """Fetch documents relevant to the user's question."""
        question = state["question"]
        # Top-k is governed by the retriever's own configuration.
        documents = retriever.invoke(question)
        return {"question": question, "documents": documents, "generation": ""}

    return retrieve_documents_node


def make_generate_node(llm: ChatGoogleGenerativeAI):
    """Build the ``generate`` node bound to a specific chat LLM.
    Args:
        llm: A LangChain chat model (Gemini Flash by default).
    Returns:
        A node callable suitable for ``StateGraph.add_node``.
    """
    # Composing prompt | llm | parser once and closing over it avoids
    # rebuilding the chain on every call.
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    rag_chain = prompt | llm | StrOutputParser()

    def generate_response_node(state: RAGGraphState) -> RAGGraphState:
        """Produce a grounded answer from the retrieved documents."""
        question = state["question"]
        documents = state["documents"]
        # Concatenate retrieved chunk text into a single context block.
        context = "\n\n".join(doc.page_content for doc in documents)
        generation = rag_chain.invoke({"context": context, "question": question})
        return {
            "question": question,
            "documents": documents,
            "generation": generation,
        }

    return generate_response_node


def build_graph(retriever, llm: ChatGoogleGenerativeAI):
    """Compile and return the ``retrieve -> generate`` LangGraph.
    Args:
        retriever: Retriever returned by ``vectorstore.as_retriever``.
        llm: Chat LLM used by the generate node.
    Returns:
        A compiled LangGraph runnable.
    """
    workflow = StateGraph(RAGGraphState)
    workflow.add_node("retrieve", make_retrieve_node(retriever))
    workflow.add_node("generate", make_generate_node(llm))
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


# ----------------------------- Main --------------------------------------- #

def main() -> None:
    """End-to-end demo: download, index, build graph, run sample queries."""
    # Fail fast if the Google key is missing.
    _require_env(*REQUIRED_ENV)

    # 1. Acquire and chunk the source text (cached after the first run).
    path = download_data()
    chunks = load_and_chunk_documents(path)

    # 2. Build the embedded vector store and a retriever over it.
    vectorstore = build_vectorstore(chunks)
    retriever = vectorstore.as_retriever()

    # 3. Build the chat LLM and compile the graph.
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    app = build_graph(retriever, llm)

    # 4. Run sample queries against the compiled graph.
    queries = (
        "What did the president say about Justice Breyer?",
        "What did the president say about the economy?",
    )
    for query in queries:
        print(f"\n--- Running RAG query: {query} ---")
        for step in app.stream({"question": query}):
            print(step)


if __name__ == "__main__":
    main()
