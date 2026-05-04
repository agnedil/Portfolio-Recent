from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.api_core import exceptions as gcp_exc

from rag_vertex.client import RagMemoryClient, RagRetrievalError, RetrievedChunk
from rag_vertex.settings import RagSettings


@pytest.fixture
def settings() -> RagSettings:
    return RagSettings(
        project_id="test-proj",
        corpus_id="test-corpus",
        similarity_top_k=3,
        vector_distance_threshold=0.7,
    )


def test_retrieve_maps_results(settings: RagSettings) -> None:
    service = MagicMock()
    service.search.return_value = [
        SimpleNamespace(text="hello", score=0.91, source_uri="gs://a"),
        SimpleNamespace(text="world", score=0.82, source_uri="gs://b"),
    ]
    client = RagMemoryClient(service=service, settings=settings)

    out = client.retrieve("a query", top_k=2)

    service.search.assert_called_once_with(query="a query", top_k=2)
    assert out == [
        RetrievedChunk(text="hello", score=0.91, source="gs://a"),
        RetrievedChunk(text="world", score=0.82, source="gs://b"),
    ]


def test_retrieve_uses_default_k_from_settings(settings: RagSettings) -> None:
    service = MagicMock()
    service.search.return_value = []
    client = RagMemoryClient(service=service, settings=settings)

    client.retrieve("q")

    service.search.assert_called_once_with(query="q", top_k=settings.similarity_top_k)


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_retrieve_rejects_blank_query(settings: RagSettings, bad: str) -> None:
    client = RagMemoryClient(service=MagicMock(), settings=settings)
    with pytest.raises(ValueError):
        client.retrieve(bad)


def test_retrieve_translates_quota_error(settings: RagSettings) -> None:
    service = MagicMock()
    service.search.side_effect = gcp_exc.ResourceExhausted("over quota")
    client = RagMemoryClient(service=service, settings=settings)

    with pytest.raises(RagRetrievalError, match="quota"):
        client.retrieve("q")


def test_retrieve_translates_permission_error(settings: RagSettings) -> None:
    service = MagicMock()
    service.search.side_effect = gcp_exc.PermissionDenied("nope")
    client = RagMemoryClient(service=service, settings=settings)

    with pytest.raises(RagRetrievalError, match="permission"):
        client.retrieve("q")
