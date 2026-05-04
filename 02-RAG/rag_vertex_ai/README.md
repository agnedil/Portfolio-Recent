# rag-vertex

A small wrapper around Google's Vertex AI RAG memory service. The Vertex AI ADK gives you `VertexAiRagMemoryService(...)` and not much else. Anything running in production needs:
- config that doesn't live in source (env vars, validated at startup),
- a typed result shape callers can depend on,
- structured error translation (quota, auth, transport),
- structured logs with latency and result counts,
- an evaluation harness so retrieval quality is a number, not a vibe.

This package is the smallest thing that does all five.

## Layout

```
src/rag_vertex/
  settings.py     # Pydantic settings, env-prefixed (RAG_*)
  client.py       # RagMemoryClient: thin wrapper, typed errors, logging
eval/
  harness.py      # recall@k harness over a YAML query set
  queries.yaml    # labeled (query, relevant_sources) pairs
tests/
  test_client.py  # unit tests against a mocked service
```

## Usage

```python
from rag_vertex import RagMemoryClient, RagSettings

client = RagMemoryClient.from_settings(RagSettings())
chunks = client.retrieve("how does the order service handle retries?")
for c in chunks:
    print(f"{c.score:.2f}  {c.source}\n  {c.text[:120]}")
```

The constructor takes an injected service, so unit tests don't need GCP credentials:

```python
client = RagMemoryClient(service=fake_service, settings=settings)
```

## Configuration

All settings load from env vars (or `.env`) with the `RAG_` prefix. See `.env.example`.
Values are validated by Pydantic at startup, so misconfiguration fails loudly rather than
at the first retrieval call.

## Evaluation

```bash
rag-eval --queries eval/queries.yaml --out eval/results.json
```

Reports `recall@5` and `recall@10` against labeled relevant sources. Tune
`RAG_SIMILARITY_TOP_K` and `RAG_VECTOR_DISTANCE_THRESHOLD` against this number, not by
eyeballing.

## End-to-end demo

`examples/answer_question.py` retrieves chunks from the corpus and pipes them into a
Gemini Flash call to actually answer the question:

```bash
rag-ask "How does the order service handle retries?"
```

Defaults to `gemini-2.5-flash`; override with `--model gemini-2.5-pro` for harder
questions. Auth reuses the same GCP project + ADC the RAG side already needs — no
separate API key. The example streams the response, refuses to answer when retrieval
returns nothing, and prints prompt / cached / output token counts to stderr so you can
see context caching engage once you wire up an explicit Gemini cache for the system
prompt.

## Notes

- **Defaults (`top_k=5`, `threshold=0.7`) are starting points, not answers.** The eval
  harness exists to tune them per corpus.

- The exact method name / response shape on `VertexAiRagMemoryService` depends on the ADK
version; the wrapper assumes a `search(query=, top_k=)` call returning objects with
`text`, `score`, and `source_uri`. Adjust the adapter in `client.py` if your SDK exposes
a different surface — that's the only place it should need to change.
