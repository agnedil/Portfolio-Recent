# Simple RAG App (LangGraph + Chroma + OpenAI / Claude / Gemini)

A minimal, production-shaped Retrieval-Augmented Generation application built
with **LangGraph**, **Chroma**, and a **Gradio** web UI that lets you switch
the answer model between **OpenAI**, **Anthropic Claude**, and **Google
Gemini** at runtime.

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
- LangGraph gives you room to grow without refactoring. The code in this
  repo is simple now, but LangGraph makes it trivial to add query rewriting,
  query routing, iterative retrieval, response evaluation, reranking,
  self-critique, multi-hop retrieval, or other agentic behaviors later: just
  call `g.add_node(...)`. For example:

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
├── llm.py               # Provider factory: OpenAI / Claude / Gemini
└── serve.py             # Gradio chat UI with a provider dropdown
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
Defines the LangGraph and owns the storage / embedding constants
(`PERSIST_DIR`, `COLLECTION_NAME`, `EMBEDDING_MODEL`, `TOP_K`) that
`index.py` imports. The graph has two nodes:
- **retrieve**: fetches the top-`k` most similar chunks from the persisted
  Chroma collection.
- **generate**: builds a prompt that constrains the answer to the retrieved
  context and calls the chat LLM passed in by the caller.

`build_graph(llm)` takes the chat LLM as an argument so the graph stays
provider-agnostic; the LLM itself is built in `llm.py` and chosen at runtime
by `serve.py`.

### `llm.py`
A small provider factory. Exposes:
- `PROVIDERS`: list of provider labels shown in the UI dropdown.
- `DEFAULT_MODELS`: default model name per provider.
- `ENV_VAR`: which environment variable each provider needs.
- `build_llm(provider, model=None, temperature=0.0)`: returns a LangChain
  `BaseChatModel` for the requested provider.

SDK imports are lazy, so users only need to install (and authenticate
against) the providers they actually intend to use.

### `serve.py`
A Gradio web app. Renders a provider dropdown above a chat interface; on
each user message it routes the question through the LangGraph compiled for
the currently selected provider. Compiled graphs are cached in-process per
provider, so switching back and forth has no rebuild cost.

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

### 3. Set API keys for the providers you plan to use

| Provider           | Environment variable | When required                              |
|--------------------|----------------------|--------------------------------------------|
| OpenAI             | `OPENAI_API_KEY`     | **Always** (used for embeddings in `index.py` and the OpenAI chat option) |
| Anthropic (Claude) | `ANTHROPIC_API_KEY`  | Only when answering with Claude            |
| Google (Gemini)    | `GOOGLE_API_KEY`     | Only when answering with Gemini            |

Example:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=AIza...
```

> **Note on embeddings:** the document index is built with OpenAI embeddings,
> so an `OPENAI_API_KEY` is always required to run `index.py` and to start
> `serve.py` (the retriever is constructed before the first question). The
> *answer* model is what switches between providers in the UI. If you would
> rather use a non-OpenAI embedding model, change `EMBEDDING_MODEL` and the
> `OpenAIEmbeddings(...)` call in `build_rag_graph.py`.

## Usage

### 1. Add your documents

Create a `docs/` directory next to the Python files and drop your text
files into it (`.txt`, `.md`, or anything that decodes as UTF-8 text):

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

### 3. Launch the chat app

```bash
python serve.py
```

Open the URL printed by Gradio (default `http://127.0.0.1:7860`). Pick a
provider from the dropdown, type your question, and read the grounded
answer plus the number of chunks used.

## Configuration

Storage and embedding constants live in `build_rag_graph.py`:

| Constant          | Default                    | Meaning                                   |
|-------------------|----------------------------|-------------------------------------------|
| `PERSIST_DIR`     | `chroma_db`                | On-disk Chroma directory                  |
| `COLLECTION_NAME` | `rag_docs`                 | Chroma collection name                    |
| `EMBEDDING_MODEL` | `text-embedding-3-small`   | OpenAI embedding model                    |
| `TOP_K`           | `4`                        | Number of chunks fetched per question     |

Provider configuration lives in `llm.py`:

| Provider           | Default model                  |
|--------------------|--------------------------------|
| OpenAI             | `gpt-4o-mini`                  |
| Anthropic (Claude) | `claude-haiku-4-5-20251001`    |
| Google (Gemini)    | `gemini-2.0-flash`             |

Edit `DEFAULT_MODELS` in `llm.py` to change the default model per provider,
or call `build_llm(provider, model="...")` directly to override.

Chunking parameters live at the top of `index.py` (`CHUNK_SIZE`,
`CHUNK_OVERLAP`, `DOCS_DIR`).

## Adding Another Provider

Adding a fourth provider is a one-file change in `llm.py`:

1. Add the label to `PROVIDERS`.
2. Add a default model entry to `DEFAULT_MODELS`.
3. Add the env-var name to `ENV_VAR`.
4. Add a branch in `build_llm` that imports the SDK and returns the chat
   model.

The dropdown picks up the new provider automatically.

## Notes

- **Errors are shown in the chat**, not the console: if you select a
  provider whose API key is missing, the chat reply will tell you exactly
  which environment variable to set.
- The vector store is local on-disk Chroma. Delete `chroma_db/` to force a
  full rebuild, or just rerun `index.py`; `Chroma.from_documents` will
  create the collection if it does not yet exist.
- The graph constrains the LLM to answer from the retrieved context only.
  If the answer is not in the documents, the model is instructed to say so.
