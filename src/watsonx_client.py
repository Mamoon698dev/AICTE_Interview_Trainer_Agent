"""
watsonx_client.py – Thin wrapper around ibm-watsonx-ai that provides
a single initialised WatsonxLLM (LangChain-compatible) and an
embeddings object backed by IBM Slate (or a local fallback).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain_ibm.embeddings import WatsonxEmbeddings

from src.config import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> WatsonxLLM:
    """Return a cached LangChain-compatible IBM Granite LLM."""
    logger.info("Initialising IBM Granite LLM: %s", config.GRANITE_MODEL_ID)
    params = {
        GenParams.MAX_NEW_TOKENS: config.MAX_NEW_TOKENS,
        GenParams.TEMPERATURE: config.TEMPERATURE,
        GenParams.TOP_P: config.TOP_P,
        GenParams.STOP_SEQUENCES: ["Human:", "User:"],
    }
    llm = WatsonxLLM(
        model_id=config.GRANITE_MODEL_ID,
        url=config.WATSONX_URL,
        apikey=config.WATSONX_API_KEY,
        project_id=config.WATSONX_PROJECT_ID,
        params=params,
    )
    return llm


@lru_cache(maxsize=1)
def get_embeddings():
    """Return a cached embeddings model.

    Primary  : IBM Slate via Watsonx (requires valid credentials).
    Fallback : HuggingFace sentence-transformers (works offline / free-tier).
    """
    missing = config.validate()
    if not missing:
        try:
            logger.info(
                "Initialising IBM Watsonx embeddings: %s",
                config.GRANITE_EMBEDDING_MODEL,
            )
            embeddings = WatsonxEmbeddings(
                model_id=config.GRANITE_EMBEDDING_MODEL,
                url=config.WATSONX_URL,
                apikey=config.WATSONX_API_KEY,
                project_id=config.WATSONX_PROJECT_ID,
            )
            # Smoke-test
            embeddings.embed_query("test")
            return embeddings
        except Exception as exc:
            logger.warning(
                "IBM Watsonx embeddings unavailable (%s). "
                "Falling back to HuggingFace sentence-transformers.",
                exc,
            )

    # ── Local fallback ────────────────────────────────────────────────────
    logger.info("Using HuggingFace embeddings (all-MiniLM-L6-v2).")
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_credentials() -> Credentials:
    """Return raw IBM Watsonx credentials (for direct API usage)."""
    return Credentials(
        url=config.WATSONX_URL,
        api_key=config.WATSONX_API_KEY,
    )
