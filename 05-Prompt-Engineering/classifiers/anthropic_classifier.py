from __future__ import annotations

import anthropic

from classifiers.base import BaseEmotionClassifier, SYSTEM_INSTRUCTION, llm_retry


_TRANSIENT = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)


class AnthropicEmotionClassifier(BaseEmotionClassifier):
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5") -> None:
        self.model = model
        self._client = anthropic.Anthropic()

    @llm_retry(_TRANSIENT)
    def classify(self, text: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=10,
            temperature=0,
            system=SYSTEM_INSTRUCTION,
            messages=[{"role": "user", "content": self.build_user_prompt(text)}],
        )
        for block in response.content:
            if block.type == "text":
                return block.text.strip()
        return ""
