from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


EMOTION_LABELS: tuple[str, ...] = (
    "Anger",
    "Disgust",
    "Fear",
    "Hope",
    "Joy",
    "Neutral",
    "Sadness",
    "Surprise",
)

SYSTEM_INSTRUCTION = "You are a very accurate text classifier."

PROMPT_TEMPLATE = (
    "Act as a very accurate zero-shot text classifier and classify the provided "
    "text into one most relevant category from the following list of categories: "
    "Sadness, Neutral, Anger, Disgust, Hope, Fear, Surprise, Joy. "
    "Classify the following text with the most relevant category from the above "
    "list and output ONLY the category and nothing else: \"{text}\""
)

RETRY_MAX_ATTEMPTS = 5
RETRY_WAIT_MAX_S = 30.0


def llm_retry(
    condition: type[BaseException]
    | tuple[type[BaseException], ...]
    | Callable[[BaseException], bool],
):
    """Tenacity decorator factory for transient LLM API errors.

    Pass either an exception class / tuple of classes (matched by isinstance), or a
    predicate callable that returns True for retryable exceptions. Retries up to
    RETRY_MAX_ATTEMPTS times with random exponential backoff capped at
    RETRY_WAIT_MAX_S seconds; the original exception is re-raised after exhaustion.
    """
    if callable(condition) and not isinstance(condition, type) and not isinstance(condition, tuple):
        retry_strategy = retry_if_exception(condition)
    else:
        retry_strategy = retry_if_exception_type(condition)
    return retry(
        retry=retry_strategy,
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        wait=wait_random_exponential(multiplier=1, max=RETRY_WAIT_MAX_S),
        reraise=True,
    )


class BaseEmotionClassifier(ABC):
    name: str = "base"
    model: str = ""

    def build_user_prompt(self, text: str) -> str:
        return PROMPT_TEMPLATE.format(text=text)

    @abstractmethod
    def classify(self, text: str) -> str:
        """Return the model's raw single-line prediction for `text`."""
