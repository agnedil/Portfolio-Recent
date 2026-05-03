"""
LLM initialization module.

Provides a single factory for the chat model used by every reasoning step
(summarization, query rewriting, agent loop, aggregation). Swap the
implementation here to plug in a different provider (Google Gemini, Anthropic,
OpenAI, etc.) without touching any other module.

Documentation:
    - https://python.langchain.com/docs/integrations/chat/ollama
    - https://python.langchain.com/docs/integrations/chat/google_generative_ai
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from config import OLLAMA_MODEL


def build_llm(temperature: float = 0.0) -> BaseChatModel:
    """Return the chat LLM used by all reasoning nodes.

    Args:
        temperature: Sampling temperature applied at construction time.

    Returns:
        A LangChain ``BaseChatModel`` ready for tool binding and invocation.

    Notes:
        To use a different provider, replace the body of this function. As long
        as the returned object implements the LangChain ``BaseChatModel``
        interface (``invoke``, ``bind_tools``, ``with_structured_output``), the
        rest of the codebase will continue to work unchanged.
    """
    # Local inference via Ollama. Swap with e.g. ChatGoogleGenerativeAI here.
    return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)
