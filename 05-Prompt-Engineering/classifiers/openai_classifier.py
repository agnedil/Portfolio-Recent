from __future__ import annotations

import openai
from openai import OpenAI

from classifiers.base import BaseEmotionClassifier, SYSTEM_INSTRUCTION, llm_retry


_TRANSIENT = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class OpenAIEmotionClassifier(BaseEmotionClassifier):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._client = OpenAI()

    @llm_retry(_TRANSIENT)
    def classify(self, text: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": self.build_user_prompt(text)},
            ],
            temperature=0,
            max_tokens=10,
        )
        return (response.choices[0].message.content or "").strip()
