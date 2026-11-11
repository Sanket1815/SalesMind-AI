"""
Centralised IBM Watsonx AI client.

Provides:
- `get_chat_model(model_id)`  → LangChain ChatWatsonx instance (for agents)
- `get_text_model(model_id)`  → ModelInference instance (for Guardian, raw generation)
- Both are cached at module level so credentials are validated once.
"""

import logging
from functools import lru_cache
from typing import Optional

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import ChatWatsonx

from config.settings import settings

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Default generation parameters per use-case
# --------------------------------------------------------------------------- #

_CHAT_PARAMS = {
    GenParams.MAX_NEW_TOKENS: 2048,
    GenParams.TEMPERATURE: 0.1,
    GenParams.TOP_P: 0.9,
    GenParams.REPETITION_PENALTY: 1.05,
}

_GUARDIAN_PARAMS = {
    GenParams.MAX_NEW_TOKENS: 20,
    GenParams.TEMPERATURE: 0.0,
    GenParams.TOP_P: 1.0,
    GenParams.DECODING_METHOD: "greedy",
}


# --------------------------------------------------------------------------- #
# Credentials helper (validated once)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _get_credentials() -> Credentials:
    if not settings.watsonx_api_key:
        raise RuntimeError(
            "WATSONX_API_KEY is not set. "
            "Copy .env.example → .env and fill in your credentials."
        )
    if not settings.watsonx_project_id:
        raise RuntimeError("WATSONX_PROJECT_ID is not set.")
    return Credentials(
        url=settings.watsonx_url,
        api_key=settings.watsonx_api_key,
    )


# --------------------------------------------------------------------------- #
# Public factories
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=8)
def get_chat_model(
    model_id: Optional[str] = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
) -> ChatWatsonx:
    """Return a cached ChatWatsonx instance for the given model."""
    model_id = model_id or settings.model_deep
    credentials = _get_credentials()
    logger.info("Initialising ChatWatsonx", extra={"model_id": model_id})
    return ChatWatsonx(
        model_id=model_id,
        url=settings.watsonx_url,
        apikey=settings.watsonx_api_key,
        project_id=settings.watsonx_project_id,
        params={
            GenParams.MAX_NEW_TOKENS: max_new_tokens,
            GenParams.TEMPERATURE: temperature,
            GenParams.TOP_P: 0.9,
            GenParams.REPETITION_PENALTY: 1.05,
        },
    )


@lru_cache(maxsize=4)
def get_text_model(model_id: Optional[str] = None) -> ModelInference:
    """Return a cached ModelInference instance (used for Guardian raw generation)."""
    model_id = model_id or settings.model_guardian
    credentials = _get_credentials()
    logger.info("Initialising ModelInference", extra={"model_id": model_id})
    return ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=settings.watsonx_project_id,
        params=_GUARDIAN_PARAMS,
    )


def get_fast_model() -> ChatWatsonx:
    return get_chat_model(model_id=settings.model_fast)


def get_deep_model() -> ChatWatsonx:
    return get_chat_model(model_id=settings.model_deep)
