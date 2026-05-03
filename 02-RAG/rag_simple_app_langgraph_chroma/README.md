# Simple RAG App (LangGraph + Chroma + OpenAI)

A minimal, production-shaped Retrieval-Augmented Generation application built
with **LangGraph**, **Chroma**, and **OpenAI**. OpenAI can be easily replaced with Claude or Gemini. Three modules, one job each:
build the index, build the graph, serve queries.

## Introduction: Why LangGraph

### 1. Explicit Control Flow
- **LangGraph**: you can see exactly how data flows between steps
  (`retrieve` -> `generate`) - makes debugging and understanding the pipeline
  much easier.
- **LangChain chains**: control flow is more implicit and hidden in chain
  internals.
- With LangGraph, it is easy to insert nodes and rewire the tree.

### 2. State Management
- **LangGraph**: explicit state object (`RAGState`) that you control.
- **Chains**: state is passed implicitly, harder to inspect or modify
  mid-pipeline.

### 3. Scalability for Complex Workflows
- LangGraph gives you room to grow without refactoring. The code in this repo
  is simple now, but LangGraph makes it trivial to add query rewriting, query
  routing, iterative retrieval, response evaluation, reranking, self-critique,
  multi-hop retrieval, or other agentic behaviors later: just call
  `g.add_node(...)`. For example:

  ```python
  g.add_node("rewrite_query", rewrite)
  g.add_node("rerank", rerank)
  g.add_node("evaluate_response", evaluate)
  ```

## Repository Layout

```
simple_rag_app/
├── README.md
├── requirements.txt
├── .gitignore
│
├── index.py             # Build the Chroma vector store from local files
├── build_rag_graph.py   # Define the retrieve -> generate LangGraph
└── serve.py             # Interactive REPL that runs the graph
```

A `docs/` directory (which you create and fill with your own files) and a
`chroma_db/` directory (created automatically by `index.py`) round out the
working tree at runtime.

## What Each File Does

### `index.py`
Loads every regular file under `docs/`, splits each document into ~800
character chunks (with 120-character overlap) using a recursive character
splitter, embeds the chunks with OpenAI's `text-embedding-3-small`, and
persists them to a local Chroma database under `chroma_db/`. Run it once
before serving the app, and again whenever your document corpus changes.

### `build_rag_graph.py`
Defines the LangGraph and owns the storage / model constants
(`PERSIST_DIR`, `COLLECTION_NAME`, `EMBEDDING_MODEL`, `CHAT_MODEL`, `TOP_K`)
that `index.py` imports. The graph has two nodes:
- **retrieve**: fetches the top-`k` most similar chunks from the persisted
  Chroma collection.
- **generate**: builds a prompt that constrains the answer to the retrieved
  context and calls `gpt-4o-mini` with `temperature=0`.

The graph is constructed inside `build_graph()`, not at import time, so
importing this module does not touch the vector store or contact OpenAI.

### `serve.py`
Compiles the graph once at startup and runs an input loop: it reads a
question, invokes the graph, prints the answer, and reports how many chunks
were used. Empty input, EOF, or `Ctrl-C` exit cleanly.

## Installation

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd simple_rag_app
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

Both the embedding step (in `index.py`) and the chat step (in
`build_rag_graph.py`) call OpenAI. Export your key in the same shell where
you will run the scripts:

```bash
export OPENAI_API_KEY=sk-...
```

## Usage

### 1. Add your documents

Create a `docs/` directory next to the Python files and drop your text files
into it (`.txt`, `.md`, or anything that decodes as UTF-8 text):

```bash
mkdir -p docs
cp /path/to/my-notes/*.md docs/
```

### 2. Build the index

```bash
python index.py
```

This loads every file in `docs/`, splits and embeds them, and writes the
Chroma database to `chroma_db/`.

### 3. Ask questions

```bash
python serve.py
```

Type a question at the `Ask:` prompt. The app prints the grounded answer and
the number of chunks the retriever returned. Press Enter on an empty line to
exit.

Sample session:

```
Ask: what does the document say about retries?
Answer:
 Retries use exponential backoff capped at 30 seconds...
Chunks used: 4
```

## Configuration

All tunable knobs live as module-level constants in `build_rag_graph.py`:

| Constant          | Default                    | Meaning                                    |
|-------------------|----------------------------|--------------------------------------------|
| `PERSIST_DIR`     | `chroma_db`                | On-disk Chroma directory                   |
| `COLLECTION_NAME` | `rag_docs`                 | Chroma collection name                     |
| `EMBEDDING_MODEL` | `text-embedding-3-small`   | OpenAI embedding model                     |
| `CHAT_MODEL`      | `gpt-4o-mini`              | OpenAI chat model                          |
| `TOP_K`           | `4`                        | Number of chunks fetched per question      |

Chunking parameters live at the top of `index.py` (`CHUNK_SIZE`,
`CHUNK_OVERLAP`, `DOCS_DIR`).

## Notes

- **`OPENAI_API_KEY` is required** for both indexing and serving.
- The vector store is local on-disk Chroma. Delete `chroma_db/` to force a
  full rebuild, or just rerun `index.py`; `Chroma.from_documents` will create
  the collection if it does not yet exist.
- The graph constrains the LLM to answer from the retrieved context only. If
  the answer is not in the documents, the model is instructed to say so.
