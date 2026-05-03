"""
LLM provider factory.

Builds a chat LLM for the selected provider so the rest of the application
stays vendor-agnostic. Each SDK is imported lazily on demand, so users only
need to install (and authenticate against) the providers they actually use.

Supported providers and required environment variables:
    - "OpenAI"             -> OPENAI_API_KEY
    - "Anthropic (Claude)" -> ANTHROPIC_API_KEY
    - "Google (Gemini)"    -> GOOGLE_API_KEY

Add another provider by extending PROVIDERS, DEFAULT_MODELS, ENV_VAR, and
the dispatch in build_llm.
"""

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

# Providers shown in the UI dropdown. Index 0 is the startup default.
PROVIDERS = ["OpenAI", "Anthropic (Claude)", "Google (Gemini)"]

# Default model per provider. Override by passing model= explicitly.
DEFAULT_MODELS = {
    "OpenAI": "gpt-4o-mini",
    "Anthropic (Claude)": "laude-sonnet-4-6",
    "Google (Gemini)": "gemini-2.0-flash",
}

# Required environment variable per provider (informational; not enforced here).
ENV_VAR = {
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic (Claude)": "ANTHROPIC_API_KEY",
    "Google (Gemini)": "GOOGLE_API_KEY",
}


def build_llm(
    provider: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Build a chat LLM for the requested provider.

    Args:
        provider: One of the entries in PROVIDERS.
        model: Optional model name; falls back to DEFAULT_MODELS[provider].
        temperature: Sampling temperature passed to the model.

    Returns:
        A LangChain ``BaseChatModel`` instance.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    # Resolve the effective model name (provider default if none given).
    model = model or DEFAULT_MODELS.get(provider)

    # Lazy imports keep the dependency surface minimal at runtime.
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    if provider == "Anthropic (Claude)":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)
    if provider == "Google (Gemini)":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    raise ValueError(
        f"Unknown provider: {provider!r}. Expected one of {PROVIDERS}."
    )
