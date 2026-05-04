from __future__ import annotations

import argparse
import sys

from google import genai
from google.genai import types

from rag_vertex.client import RagMemoryClient, RetrievedChunk
from rag_vertex.settings import RagSettings


SYSTEM_PROMPT = """You answer questions using only the provided <context>.

Rules:
- Cite the source after each claim as [n] where n is the chunk number.
- If the context does not contain the answer, say so plainly. Do not speculate.
- Be concise. Prefer 2-3 sentences unless the question demands depth.
"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        src = c.source or "unknown"
        blocks.append(f"[{i}] source={src} score={c.score:.2f}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def answer(question: str, *, model: str = "gemini-2.5-flash", top_k: int = 5) -> str:
    settings = RagSettings()
    rag = RagMemoryClient.from_settings(settings)
    chunks = rag.retrieve(question, top_k=top_k)

    if not chunks:
        msg = "(no relevant context retrieved — refusing to answer from prior knowledge)"
        print(msg)
        return msg

    client = genai.Client(
        vertexai=True,
        project=settings.project_id,
        location=settings.location,
    )
    user_content = f"<context>\n{format_context(chunks)}\n</context>\n\n{question}"
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=1024,
    )

    parts: list[str] = []
    final_chunk = None
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=user_content,
        config=config,
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)
            parts.append(chunk.text)
        final_chunk = chunk
    print()

    if final_chunk is not None and final_chunk.usage_metadata is not None:
        u = final_chunk.usage_metadata
        cached = u.cached_content_token_count or 0
        print(
            f"\n[usage] prompt={u.prompt_token_count} "
            f"cached={cached} "
            f"output={u.candidates_token_count}",
            file=sys.stderr,
        )

    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question against the Vertex AI RAG corpus, answered by Gemini Flash."
    )
    parser.add_argument("question", nargs="+", help="The question to ask.")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    answer(" ".join(args.question), model=args.model, top_k=args.top_k)


if __name__ == "__main__":
    main()
