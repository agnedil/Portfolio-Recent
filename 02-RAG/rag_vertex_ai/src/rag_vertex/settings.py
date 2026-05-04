from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_id: str
    corpus_id: str
    location: str = "us-central1"

    similarity_top_k: int = Field(default=5, ge=1, le=50)
    vector_distance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    request_timeout_s: float = Field(default=10.0, gt=0.0)

    @property
    def corpus_resource_name(self) -> str:
        return (
            f"projects/{self.project_id}"
            f"/locations/{self.location}"
            f"/ragCorpora/{self.corpus_id}"
        )
