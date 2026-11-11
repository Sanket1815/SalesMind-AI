import logging
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # IBM Watsonx credentials
    watsonx_api_key: str = ""
    watsonx_project_id: str = ""
    watsonx_url: str = "https://us-south.ml.cloud.ibm.com"

    # Model IDs
    model_fast: str = "ibm/granite-3-3-8b-instruct"
    model_deep: str = "ibm/granite-4.1-8b-instruct"
    model_guardian: str = "ibm/granite-guardian-3-8b"

    # Qualification thresholds
    qualification_threshold: int = 60   # score >= this → qualified
    confidence_threshold: float = 0.75  # confidence >= this → use fast model

    # Guardian retry budget
    guardian_max_retries: int = 2

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # PostgreSQL
    database_url: Optional[str] = None
    # e.g. postgresql+asyncpg://salesmind:secret@localhost:5432/salesmind

    # Optional enhanced web search
    tavily_api_key: Optional[str] = None

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "console"

    # LangGraph persistence URI (falls back to SQLite when no Postgres)
    langgraph_checkpoint_uri: str = "sqlite:///checkpoints/pipeline.db"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        numeric = getattr(logging, v.upper(), None)
        if not isinstance(numeric, int):
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()

    @field_validator("qualification_threshold")
    @classmethod
    def validate_threshold(cls, v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError("qualification_threshold must be 0-100")
        return v

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_threshold must be 0.0-1.0")
        return v


settings = Settings()
