"""Embedding model configuration — never rely on LlamaIndex OpenAI defaults."""

from __future__ import annotations

from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings import MockEmbedding

from chatbot.config import AppConfig, DEFAULT_CONFIG

_CONFIGURED = False


def mock_embed_model(embed_dim: int = 8) -> MockEmbedding:
    return MockEmbedding(embed_dim=embed_dim)


def configure_embed_model(
    config: AppConfig | None = None,
    *,
    for_testing: bool = False,
) -> BaseEmbedding:
    global _CONFIGURED
    if for_testing:
        model: BaseEmbedding = mock_embed_model()
    else:
        cfg = config or DEFAULT_CONFIG
        if cfg.embed_provider == "huggingface":
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            model = HuggingFaceEmbedding(model_name=cfg.hf_embed_model)
        else:
            from llama_index.embeddings.ollama import OllamaEmbedding

            model = OllamaEmbedding(
                model_name=cfg.embed_model,
                base_url=cfg.ollama_base_url,
            )
    Settings.embed_model = model
    _CONFIGURED = True
    return model


def require_embed_model() -> BaseEmbedding:
    """Return the active embed model; fail fast instead of defaulting to OpenAI."""
    if not _CONFIGURED:
        raise RuntimeError(
            "Embedding model is not configured. Call configure_embed_model() or "
            "configure_llama_index() before building or querying an index."
        )
    return Settings.embed_model
