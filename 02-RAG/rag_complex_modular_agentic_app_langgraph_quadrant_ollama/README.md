# Agentic RAG Application

A production-grade, modular implementation of an agentic Retrieval-Augmented
Generation (RAG) system over PDF documents. Built with **LangGraph**,
**LangChain**, **Qdrant** (hybrid dense + sparse), **HuggingFace** sentence
transformers, **FastEmbed** BM25, and **Gradio**.

The system converts PDFs to Markdown, applies a parent / child chunking
strategy, indexes the chunks into a hybrid vector store, and serves a chat
interface backed by a LangGraph agent that searches, retrieves parent context
on demand, retries on empty results, splits multi-part questions into parallel
sub-agents, and aggregates the answers into a single response.

## Architecture

```
PDF documents
   |  text_processing.py  (PyMuPDF -> Markdown)
   v
Markdown files
   |  chunking.py         (parent / child hierarchy)
   v
Parent JSONs   +   Child Documents
                              |  embeddings.py  (dense + sparse)
                              v
                       Qdrant collection (hybrid)

User query (Gradio)
   |  app.py
   v
LangGraph orchestrator (graph.py + nodes.py)
   |
   +--> summarize  --> rewrite  --> [human_input | parallel agents]
                                                       |
                                                       v
                                            ReAct agent subgraph
                                                  |
                                            search_child_chunks
                                                  |
                                            retrieve_parent_chunks (optional)
                                                  |
                                            answer
                                                  |
                                       aggregate_responses
                                                  |
                                            final answer
```

## Repository Layout

```
agentic_rag_app/
├── README.md
├── requirements.txt
├── .gitignore
│
├── config.py            # Paths, model names, chunking + retrieval parameters
├── llm.py               # Chat LLM factory (swap provider here)
├── embeddings.py        # Dense + sparse embedding factories
├── vector_store.py      # Qdrant client and hybrid collection lifecycle
├── text_processing.py   # PDF -> Markdown conversion
├── chunking.py          # Parent / child hierarchical chunking strategy
├── indexing.py          # Per-document indexing pipeline
├── tools.py             # Agent retrieval tools (search + parent fetch)
├── prompts.py           # System prompts for every reasoning node
├── state.py             # LangGraph state schemas + reducers
├── nodes.py             # Graph node implementations
├── graph.py             # LangGraph wiring (outer + inner subgraph)
├── app.py               # Gradio chat interface
│
├── ingest.py            # Entry point: build the vector store from PDFs
└── serve.py             # Entry point: launch the chat application
```

Local data directories (created on first run, ignored by git):

```
docs/          # Drop your source PDFs here
markdown/      # Auto-generated Markdown
parent_store/  # Parent chunks as one JSON file each
qdrant_db/     # On-disk Qdrant database
```

## Installation

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd agentic_rag_app
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install Ollama and pull the model

The default LLM is served locally via [Ollama](https://ollama.com/). Install
Ollama for your platform, start the daemon, then pull the model used by
`config.py`:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

You can use any other Ollama-compatible model by setting the `OLLAMA_MODEL`
environment variable, or by editing `llm.py` to use a different provider
(e.g., Google Gemini, Anthropic, OpenAI). See **Swapping Components** below.

## Usage

### 1. Add your PDFs

Place one or more `.pdf` files into `docs/` (the directory is created
automatically on first run).

### 2. Build the index

```bash
python ingest.py
```

This converts PDFs to Markdown, resets the Qdrant collection, chunks every
document into parent / child pairs, and writes the children to Qdrant and the
parents to `parent_store/` as JSON.

Re-run `ingest.py` whenever your PDF corpus changes.

### 3. Launch the chat application

```bash
python serve.py
```

The Gradio app starts on `http://127.0.0.1:7860` by default. Override with
`GRADIO_SERVER_NAME` and `GRADIO_SERVER_PORT` environment variables.

## What Each Module Does When You Run It

| Module                | Run as a script?  | Behavior |
|-----------------------|-------------------|----------|
| `ingest.py`           | Yes (entry point) | Converts PDFs to Markdown, resets the Qdrant collection, indexes child chunks, persists parent chunks as JSON. Idempotent: skips already-converted Markdown. |
| `serve.py`            | Yes (entry point) | Connects to the existing vector store, builds the LLM and tools, compiles the LangGraph, and launches the Gradio chat UI. Does **not** re-index. |
| `text_processing.py`  | Importable        | Provides `pdfs_to_markdowns()` and `pdf_to_markdown()`. |
| `chunking.py`         | Importable        | Provides `build_parent_child_chunks()` plus the merge / split / clean helpers. |
| `embeddings.py`       | Importable        | Builds dense + sparse embedding objects and exposes the dense embedding dimension. |
| `vector_store.py`     | Importable        | Builds the Qdrant client, ensures or resets a hybrid collection, and returns a `QdrantVectorStore` for hybrid retrieval. |
| `indexing.py`         | Importable        | Reads `markdown/`, chunks each document, writes children to the vector store, and persists parent JSONs. |
| `tools.py`            | Importable        | Returns two `@tool` callables: `search_child_chunks`, `retrieve_parent_chunks`. |
| `llm.py`              | Importable        | Returns a configured chat LLM (Ollama by default). |
| `prompts.py`          | Importable        | Returns the four system prompts used by graph nodes. |
| `state.py`            | Importable        | Defines `State`, `AgentState`, `QueryAnalysis`, and the `accumulate_or_reset` reducer. |
| `nodes.py`            | Importable        | Builds the seven node functions that implement the workflow. |
| `graph.py`            | Importable        | Wires the inner ReAct subgraph and outer orchestrator graph; returns a compiled graph with checkpointer + interrupt. |
| `app.py`              | Importable        | Builds the Gradio Blocks demo and exposes a `launch()` helper used by `serve.py`. |
| `config.py`           | Importable        | Provides constants and `ensure_directories()`. |

## Swapping Components

Each module is a single replaceable concern. To swap any component, change
**one** module and leave the rest untouched.

| To replace ...                     | Edit ...           | What to change |
|------------------------------------|--------------------|---------------|
| The chat LLM (Gemini, Anthropic, OpenAI, vLLM) | `llm.py`           | Replace the body of `build_llm()` with the new provider. As long as it returns a `BaseChatModel`, `bind_tools` and `with_structured_output` continue to work. |
| Dense or sparse embedding models   | `embeddings.py`    | Change the model names in `build_dense_embeddings` / `build_sparse_embeddings`, or rewrite to use a different embeddings provider. |
| The vector database                | `vector_store.py`  | Replace the Qdrant client and `QdrantVectorStore` with a different LangChain `VectorStore` (Pinecone, Weaviate, pgvector, Milvus). |
| Document loader / parser           | `text_processing.py` | Replace the PyMuPDF pipeline with Unstructured, Docling, OCR, or a HTML / DOCX-specific loader. The contract is: produce `.md` files in `MARKDOWN_DIR`. |
| Chunking strategy                  | `chunking.py`      | Replace `build_parent_child_chunks` with a semantic chunker, late chunking, fixed windows, etc. The function must return `(parent_pairs, children)`. |
| Persistence for parent chunks      | `indexing.py` and `tools.py` | Swap the JSON-on-disk store for Redis / S3 / SQLite; update `_save_parent_chunks` and `retrieve_parent_chunks` accordingly. |
| Agent tools                        | `tools.py`         | Add or remove `@tool` callables. The agent will discover them via `bind_tools`. |
| Prompts                            | `prompts.py`       | Edit prompt text without touching wiring. |
| Graph topology (e.g., add a critic node, reflection loop) | `graph.py` (and `nodes.py` for new node implementations) | Add nodes / edges. Existing nodes keep working as long as state schemas are unchanged. |
| State schema                       | `state.py`         | Update reducers and field types. Touch `nodes.py` only if a node depends on the changed field. |
| Web UI                             | `app.py`           | Swap Gradio for FastAPI, Streamlit, Slack, etc. The contract is: call the compiled graph object. |

## Configuration via Environment Variables

All paths, models, chunking parameters, and the Gradio server address are
configurable via environment variables. See `config.py` for the full list and
defaults. Examples:

```bash
export DOCS_DIR=/data/my-pdfs
export OLLAMA_MODEL=llama3.1:8b-instruct
export DENSE_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
export SEARCH_SCORE_THRESHOLD=0.6
export GRADIO_SERVER_PORT=8080
```

## Notes and Caveats

- **Ollama must be running** before `serve.py` is launched. Verify with
  `ollama list` that the configured model is available.
- **`InMemorySaver` is used for short-term memory.** Conversation state lives
  inside the running process; restarting the server clears all threads. Swap
  in a persistent checkpointer (e.g., `SqliteSaver`, `PostgresSaver`) inside
  `graph.py` if you need durability.
- **`reset_collection` in `ingest.py` drops the existing collection.** If you
  want incremental indexing instead of full rebuilds, replace it with
  `ensure_collection` and adapt `indexing.py` to upsert by stable IDs.
- **Hybrid retrieval** combines dense cosine similarity with BM25; the
  `score_threshold` filter is applied to the fused score. Tune
  `SEARCH_SCORE_THRESHOLD` if the agent reports too many or too few results.
