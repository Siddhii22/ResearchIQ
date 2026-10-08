"""Environment-backed application settings."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Settings loaded from environment variables and an optional .env file."""

    app_name: str = os.getenv("APP_NAME", "ResearchIQ")
    llm_api_key: str | None = os.getenv("LLM_API_KEY")


settings = Settings()