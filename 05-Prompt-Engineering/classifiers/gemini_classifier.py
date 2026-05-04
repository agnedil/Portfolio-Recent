from __future__ import annotations

from google import genai
from google.api_core import exceptions as gcp_exc
from google.genai import types

from classifiers.base import BaseEmotionClassifier, SYSTEM_INSTRUCTION, llm_retry


_TRANSIENT = (
    gcp_exc.ResourceExhausted,
    gcp_exc.ServiceUnavailable,
    gcp_exc.DeadlineExceeded,
    gcp_exc.InternalServerError,
)


class GeminiEmotionClassifier(BaseEmotionClassifier):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        self.model = model
        self._client = genai.Client()

    @llm_retry(_TRANSIENT)
    def classify(self, text: str) -> str:
        response = self._client.models.generate_content(
            model=self.model,
            contents=self.build_user_prompt(text),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0,
                max_output_tokens=10,
            ),
        )
        return (response.text or "").strip()
