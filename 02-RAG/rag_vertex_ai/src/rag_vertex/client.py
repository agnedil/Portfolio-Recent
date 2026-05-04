from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from google.api_core import exceptions as gcp_exc

from rag_vertex.settings import RagSettings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    source: str | None


class _MemoryService(Protocol):
    def search(self, *, query: str, top_k: int) -> list[Any]: ...


class RagRetrievalError(RuntimeError):
    pass


class RagMemoryClient:
    def __init__(self, service: _MemoryService, settings: RagSettings) -> None:
        self._service = service
        self._settings = settings

    @classmethod
    def from_settings(cls, settings: RagSettings) -> "RagMemoryClient":
        from google.adk.memory import VertexAiRagMemoryService

        service = VertexAiRagMemoryService(
            rag_corpus=settings.corpus_resource_name,
            similarity_top_k=settings.similarity_top_k,
            vector_distance_threshold=settings.vector_distance_threshold,
        )
        return cls(service=service, settings=settings)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        k = top_k if top_k is not None else self._settings.similarity_top_k
        started = time.perf_counter()

        try:
            raw = self._service.search(query=query, top_k=k)
        except gcp_exc.PermissionDenied as e:
            log.error("rag.retrieve.permission_denied", extra={"query_len": len(query)})
            raise RagRetrievalError("permission denied for RAG corpus") from e
        except gcp_exc.ResourceExhausted as e:
            log.warning("rag.retrieve.quota_exhausted", extra={"query_len": len(query)})
            raise RagRetrievalError("RAG quota exhausted") from e
        except gcp_exc.GoogleAPIError as e:
            log.exception("rag.retrieve.api_error")
            raise RagRetrievalError(f"RAG call failed: {e}") from e

        chunks = [
            RetrievedChunk(
                text=getattr(r, "text", ""),
                score=float(getattr(r, "score", 0.0)),
                source=getattr(r, "source_uri", None),
            )
            for r in raw
        ]
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        log.info(
            "rag.retrieve.ok",
            extra={
                "query_len": len(query),
                "k": k,
                "n_chunks": len(chunks),
                "latency_ms": latency_ms,
            },
        )
        return chunks
