from __future__ import annotations

import os

import requests
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from classifiers.base import BaseEmotionClassifier, SYSTEM_INSTRUCTION, llm_retry


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, HfHubHTTPError):
        resp = getattr(exc, "response", None)
        if resp is None:
            return False
        sc = resp.status_code
        return sc in (408, 429) or sc >= 500
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))


class HuggingFaceEmotionClassifier(BaseEmotionClassifier):
    name = "huggingface"

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        provider: str | None = None,
    ) -> None:
        self.model = model
        self._client = InferenceClient(
            model=model,
            provider=provider,
            token=os.environ.get("HF_TOKEN"),
        )

    @llm_retry(_is_transient)
    def classify(self, text: str) -> str:
        response = self._client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": self.build_user_prompt(text)},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        content = response.choices[0].message.content or ""
        return content.strip()
