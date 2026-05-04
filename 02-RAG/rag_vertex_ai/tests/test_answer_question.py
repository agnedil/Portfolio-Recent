from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from examples import answer_question
from rag_vertex.client import RetrievedChunk


@pytest.fixture
def fake_settings() -> SimpleNamespace:
    return SimpleNamespace(project_id="test-proj", location="us-central1")


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch, fake_settings: SimpleNamespace) -> SimpleNamespace:
    monkeypatch.setattr(
        "examples.answer_question.RagSettings",
        lambda: fake_settings,
    )
    fake_rag = MagicMock()
    monkeypatch.setattr(
        "examples.answer_question.RagMemoryClient.from_settings",
        lambda settings: fake_rag,
    )
    fake_genai = MagicMock()
    genai_ctor = MagicMock(return_value=fake_genai)
    monkeypatch.setattr("examples.answer_question.genai.Client", genai_ctor)
    return SimpleNamespace(rag=fake_rag, genai=fake_genai, genai_ctor=genai_ctor)


def _text_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, usage_metadata=None)


def _final_chunk() -> SimpleNamespace:
    return SimpleNamespace(
        text=None,
        usage_metadata=SimpleNamespace(
            prompt_token_count=120,
            candidates_token_count=42,
            cached_content_token_count=0,
        ),
    )


def test_empty_retrieval_skips_gemini(
    patched: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    patched.rag.retrieve.return_value = []

    out = answer_question.answer("anything?")

    assert "no relevant context" in out
    patched.genai_ctor.assert_not_called()


def test_happy_path_streams_and_reports_usage(
    patched: SimpleNamespace,
    fake_settings: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    patched.rag.retrieve.return_value = [
        RetrievedChunk(
            text="Order service uses exponential backoff.",
            score=0.91,
            source="gs://docs/orders.md",
        ),
        RetrievedChunk(
            text="Max 5 retries.",
            score=0.82,
            source="gs://docs/retries.md",
        ),
    ]
    patched.genai.models.generate_content_stream.return_value = iter(
        [
            _text_chunk("Exponential backoff "),
            _text_chunk("with up to 5 retries [1][2]."),
            _final_chunk(),
        ]
    )

    out = answer_question.answer(
        "How does the order service handle retries?",
        model="gemini-2.5-flash",
    )

    assert out == "Exponential backoff with up to 5 retries [1][2]."

    patched.genai_ctor.assert_called_once_with(
        vertexai=True,
        project=fake_settings.project_id,
        location=fake_settings.location,
    )

    call = patched.genai.models.generate_content_stream.call_args
    assert call.kwargs["model"] == "gemini-2.5-flash"

    contents = call.kwargs["contents"]
    assert "[1] source=gs://docs/orders.md" in contents
    assert "[2] source=gs://docs/retries.md" in contents
    assert "Order service uses exponential backoff" in contents
    assert "How does the order service handle retries?" in contents

    config = call.kwargs["config"]
    assert "answer questions using only the provided" in config.system_instruction
    assert config.max_output_tokens == 1024

    captured = capsys.readouterr()
    assert "Exponential backoff with up to 5 retries [1][2]." in captured.out
    assert "prompt=120" in captured.err
    assert "output=42" in captured.err
    assert "cached=0" in captured.err
