# RAG

A collection of Retrieval-Augmented Generation (RAG) examples that implement the same core pattern — *retrieve relevant context, then generate an answer grounded in it* — across different vector stores, orchestration frameworks, and LLM providers. The directory progresses from a single-file LangChain pipeline up to a fully modular, agentic LangGraph application.

## Introduction: Why several vector stores, frameworks, and LLMs?

RAG is not a single product — it is a pattern, and every step of that pattern (storage, orchestration, generation) has multiple mature implementations with different trade-offs. The examples here deliberately mix and match them so the same pattern can be compared side by side.

### Why several vector stores (Chroma, FAISS, Weaviate, Qdrant, Vertex AI RAG Corpus)?

- **FAISS** — an in-process library, fastest to spin up, no server, ideal for prototypes and small/medium corpora that fit in RAM.
- **Chroma** — an embedded, file-backed store with a simple Python API; a good default for local development and persisted small projects.
- **Weaviate** — a full vector database with a schema, hybrid search, and an embedded mode; useful when you need richer metadata filtering or a server-style deployment but still want one-process simplicity.
- **Qdrant** — a production-grade vector DB with strong support for hybrid (dense + sparse / BM25) retrieval, payload filtering, and on-disk collections that scale beyond memory.
- **Vertex AI RAG Corpus** — Google Cloud's managed RAG service; the right choice when you want storage, embedding, and retrieval to be operated for you inside GCP with IAM, quotas, and audit built in.

In short: prototypes lean toward FAISS/Chroma, hybrid retrieval and richer filtering pull you toward Qdrant/Weaviate, and managed/governed deployments lean toward Vertex AI.

### Why several frameworks (LangChain, LangGraph, CrewAI)?

- **LangChain** is great for *linear* RAG: a chain of retrieve → (rerank) → generate is concise and readable. It hits its limits when you want to inspect or modify intermediate state, add cycles (retry, self-critique), or branch the flow.
- **LangGraph** models the pipeline as an explicit state graph. Control flow and state are first-class, which makes it easy to add nodes (query rewriting, reranking, evaluation, multi-hop retrieval) without refactoring, and easy to debug because every transition is visible.
- **CrewAI** layers a *multi-agent* abstraction on top — separate agents (router, retriever, grader, hallucination checker, final answerer) cooperate on a task. This shines for agentic RAG where each step has a distinct role and prompt.

### Why several LLMs (Gemini, Llama, GPT, Claude, Qwen, etc.)?

Different LLMs are picked for different reasons: cost, latency, privacy/locality, context length, and licensing. The examples cover the main deployment modes:

- **Hosted closed models** — `gpt-4o-mini` (OpenAI), `claude-haiku-4-5` (Anthropic), `gemini-2.0-flash` / `gemini-2.5-flash` (Google). Strongest general quality, easiest API, no infrastructure to run.
- **Hosted open-weights models** — `llama3-8b-8192` via Groq, `meta-llama-3-70b` via Replicate. Open weights served by a third-party inference provider; usually cheaper or faster than the closed frontier models.
- **Locally hosted open-weights models** — `qwen3:4b-instruct` via Ollama. Runs on your own hardware, keeps data in-house, no per-token cost.
- **Managed enterprise** — Google Vertex AI's RAG memory service paired with Gemini. Both retrieval and generation live inside the cloud boundary.

Showing the same pattern across providers makes it cheap to swap one piece (the LLM, the store, the framework) without rewriting the whole pipeline. Several apps in this folder are explicitly designed to make that swap a one-file change.

## Repository layout

### Root-level Python files

- **`rag_advanced_langchain_faiss_llama.py`** — Single-file *advanced* LangChain RAG. Loads PDFs from URL, splits with `RecursiveCharacterTextSplitter`, builds **hybrid retrieval** (BM25 + FAISS over Cohere embeddings) wrapped in an `EnsembleRetriever` (weights `[0.6, 0.4]`), applies a **Cohere reranker**, and answers conversationally with **Llama-3-70B via Replicate** using `ConversationalRetrievalChain` with chat history. Demonstrates how far you can push pure LangChain *chains* before reaching for a graph.

- **`rag_agentic_app_crewai_llama.py`** — Multi-agent RAG packaged as a **Gradio** web app. Uses **CrewAI** to coordinate five agents (Router → Retriever → Grader → Hallucination Grader → Final Answer) over a `PDFSearchTool` (vector store over the "Attention Is All You Need" PDF) and a `TavilySearchResults` web-search tool, all driven by **Llama-3-8B on Groq**. The Router decides between vectorstore lookup and web search; the Hallucination Grader checks factual grounding before the final synthesis.

- **`rag_langgraph_weaviate_gemini.py`** — Minimal **LangGraph + Weaviate + Gemini** pipeline. Downloads a text document, chunks it with `CharacterTextSplitter`, indexes it into an **embedded Weaviate** instance using **Google Generative AI embeddings**, and answers via a two-node graph (`retrieve → generate`) backed by **Gemini 2.0 Flash**. The simplest demonstration of "chain → graph" with a non-OpenAI embedding/LLM stack.

### Subfolders (each is a self-contained mini-project with its own README)

- **`rag_simple_app_langgraph_chroma/`** — *Three-file* production-shaped LangGraph + Chroma + OpenAI (`gpt-4o-mini`) app. Strict separation of concerns: `index.py` builds the on-disk Chroma store from `docs/`, `build_rag_graph.py` defines the `retrieve → generate` graph and owns all storage/model constants, `serve.py` runs an interactive REPL. The reference implementation for "what a clean LangGraph RAG looks like."

- **`rag_simple_app_langgraph_chroma_select_llm_provider/`** — Same shape as the previous app, plus a **Gradio UI with a provider dropdown** that swaps the answer model between **OpenAI**, **Anthropic Claude**, and **Google Gemini** at runtime. `llm.py` is a small provider factory with lazy SDK imports; compiled graphs are cached per provider so switching has no rebuild cost. Adding a fourth provider is a one-file change.

- **`rag_complex_modular_agentic_app_langgraph_quadrant_ollama/`** — The most advanced example: a **production-grade modular agentic RAG** over PDF documents. PDFs are converted to Markdown (PyMuPDF), parent/child chunked, indexed into **Qdrant** with **hybrid retrieval** (dense HuggingFace `all-mpnet-base-v2` + sparse FastEmbed BM25), and served via a **Gradio** chat backed by a **LangGraph** orchestrator with a ReAct sub-agent, parallel sub-agents for multi-part questions, parent-context retrieval on demand, and an `InMemorySaver` checkpointer. LLM defaults to a **local Ollama Qwen** model. Every concern (`config.py`, `llm.py`, `embeddings.py`, `vector_store.py`, `chunking.py`, `tools.py`, `prompts.py`, `state.py`, `nodes.py`, `graph.py`, `app.py`) is its own file, so any single piece can be swapped without touching the others. Two entry points: `ingest.py` (build the index) and `serve.py` (run the chat).

- **`rag_vertex_ai/`** — A small Python package that wraps Google's **Vertex AI RAG memory service** with the things production needs but the raw ADK does not provide: env-validated Pydantic settings (`RAG_*`), a typed `RagMemoryClient` with structured error translation and logging, an evaluation harness that reports `recall@5` / `recall@10` against a labeled `queries.yaml`, unit tests against a mocked service, and an end-to-end `rag-ask` CLI that streams a **Gemini 2.5 Flash** answer over the retrieved chunks. The "what a managed RAG looks like in real code" example.

- **`ynotebooks/`**
  - `RAG_LangChain_Vs_LangGraph.ipynb` — A side-by-side notebook walking through the same RAG task implemented first as a LangChain chain and then as a LangGraph graph, with notes on when to prefer one over the other (control flow, explicit state, room to grow).

## Suggested reading order

1. **`rag_advanced_langchain_faiss_llama.py`** — the simplest end-to-end pipeline, in pure LangChain.
2. **`ynotebooks/RAG_LangChain_Vs_LangGraph.ipynb`** — explicit chain-vs-graph comparison.
3. **`rag_simple_app_langgraph_chroma/`** — same idea split into clean modules.
4. **`rag_simple_app_langgraph_chroma_select_llm_provider/`** — add a provider switch and a Gradio UI.
5. **`rag_langgraph_weaviate_gemini.py`** — swap the store (Weaviate) and the LLM stack (Gemini).
6. **`rag_agentic_app_crewai_llama.py`** — first taste of multi-agent RAG with CrewAI.
7. **`rag_complex_modular_agentic_app_langgraph_quadrant_ollama/`** — full agentic LangGraph with hybrid Qdrant retrieval, parent/child chunking, and a local LLM.
8. **`rag_vertex_ai/`** — what RAG looks like as a managed cloud service with proper config, errors, logging, and evaluation.
