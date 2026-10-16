"""Connection settings, loaded from .env.

No model here on purpose: the model is chosen per request (OpenrouterConfig).
This holds only connection and secret values.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    http_referer: str | None = None
    x_title: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # values come from env/.env
