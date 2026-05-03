"""
Gradio chat UI for the RAG application with provider switching.

Lets the user pick OpenAI, Claude, or Gemini at runtime via a dropdown above
the chatbot. A graph is compiled lazily the first time each provider is used
and cached in-process, so subsequent turns reuse the same retriever and
chat model.

Usage:
    # Set keys for whichever providers you intend to use.
    # OPENAI_API_KEY is always required because embeddings use OpenAI.
    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...     # only if using Claude
    export GOOGLE_API_KEY=...        # only if using Gemini
    python serve.py
"""

import logging
from typing import Dict
import gradio as gr

from build_rag_graph import build_graph
from llm import DEFAULT_MODELS, ENV_VAR, PROVIDERS, build_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Compiled-graph cache keyed by provider name. Built lazily on first use.
_graph_cache: Dict[str, object] = {}


def get_graph_for(provider: str):
    """Return the compiled graph for the given provider, building it if needed."""
    if provider not in _graph_cache:
        logger.info(
            "Building graph for provider %r (model=%s)",
            provider, DEFAULT_MODELS.get(provider),
        )
        # Construct the chat LLM and compile the graph once per provider.
        llm = build_llm(provider)
        _graph_cache[provider] = build_graph(llm)
    return _graph_cache[provider]


def chat_fn(message: str, history, provider: str) -> str:
    """Single-turn handler: route the message to the selected provider's graph."""
    msg = (message or "").strip()
    if not msg:
        return ""

    try:
        rag_app = get_graph_for(provider)
        result = rag_app.invoke({"question": msg, "docs": [], "answer": ""})
        return f"{result['answer']}\n\n_Chunks used: {len(result['docs'])}_"
    except Exception as e:
        # Surface common errors (missing API key, bad model name) to the chat.
        env = ENV_VAR.get(provider, "")
        return (
            f"**Error from {provider}:** {e}\n\n"
            f"Make sure the environment variable `{env}` is set and the "
            f"corresponding SDK package is installed."
        )


def main() -> None:
    """Build and launch the Gradio app."""
    with gr.Blocks(title="Simple RAG") as demo:
        gr.Markdown(
            "# Simple RAG\n"
            "Ask questions grounded in the documents you indexed with "
            "`python index.py`. Pick a provider for the answer model:"
        )
        # Provider dropdown rendered above the chat; re-used as ChatInterface input.
        provider = gr.Dropdown(
            choices=PROVIDERS,
            value=PROVIDERS[0],
            label="Provider",
        )
        gr.ChatInterface(
            fn=chat_fn,
            additional_inputs=[provider],
            type="messages",
        )

    demo.launch()


if __name__ == "__main__":
    main()
