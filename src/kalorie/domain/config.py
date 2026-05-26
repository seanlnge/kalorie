from pathlib import Path
from typing import Any, Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

RunMode = Literal["local", "api", "kalshi_public", "kalshi_authorized"]


class Settings(BaseSettings):
    api_ninjas_api_key: SecretStr | None = None
    financial_modeling_prep_api_key: SecretStr | None = None
    newsdata_api_key: SecretStr | None = None
    tiingo_api_key: SecretStr | None = None
    defeat_beta_api_key: SecretStr | None = None
    sec_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int | None = Field(default=None, gt=0)
    embedding_batch_size: int = Field(default=100, gt=0)
    embedding_cache_path: Path = Path(".kalorie-cache/embeddings.json")
    kalshi_api_key_id: SecretStr | None = None
    kalshi_private_key_path: Path | None = None
    kalshi_base_url: AnyHttpUrl = "https://api.elections.kalshi.com/trade-api/v2"
    mode: RunMode = "local"

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(*args, **kwargs)
        data["api_ninjas_api_key"] = "**********" if self.api_ninjas_api_key else None
        data["financial_modeling_prep_api_key"] = (
            "**********" if self.financial_modeling_prep_api_key else None
        )
        data["newsdata_api_key"] = "**********" if self.newsdata_api_key else None
        data["tiingo_api_key"] = "**********" if self.tiingo_api_key else None
        data["defeat_beta_api_key"] = "**********" if self.defeat_beta_api_key else None
        data["sec_api_key"] = "**********" if self.sec_api_key else None
        data["openai_api_key"] = "**********" if self.openai_api_key else None
        data["kalshi_api_key_id"] = "**********" if self.kalshi_api_key_id else None
        return data

    def redacted_dict(self) -> dict[str, Any]:
        return {
            "api_ninjas_configured": self.api_ninjas_api_key is not None,
            "financial_modeling_prep_configured": self.financial_modeling_prep_api_key is not None,
            "newsdata_configured": self.newsdata_api_key is not None,
            "tiingo_configured": self.tiingo_api_key is not None,
            "defeat_beta_configured": self.defeat_beta_api_key is not None,
            "sec_api_configured": self.sec_api_key is not None,
            "kalshi_api_key_id_configured": self.kalshi_api_key_id is not None,
            "kalshi_private_key_path_configured": self.kalshi_private_key_path is not None,
            "openai_configured": self.openai_api_key is not None,
            "openai_embedding_model": self.openai_embedding_model,
            "openai_embedding_dimensions": self.openai_embedding_dimensions,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_cache_path": self.embedding_cache_path.as_posix(),
            "kalshi_base_url": str(self.kalshi_base_url).rstrip("/"),
            "mode": self.mode,
        }

    def validate_for_mode(self, mode: RunMode | None = None) -> None:
        selected_mode = mode or self.mode
        if selected_mode == "kalshi_authorized":
            if self.kalshi_api_key_id is None:
                raise ValueError("KALSHI_API_KEY_ID is required for kalshi_authorized mode")
            if self.kalshi_private_key_path is None:
                raise ValueError("KALSHI_PRIVATE_KEY_PATH is required for kalshi_authorized mode")
            if not self.kalshi_private_key_path.exists():
                raise ValueError(
                    f"KALSHI_PRIVATE_KEY_PATH does not exist: {self.kalshi_private_key_path}"
                )
