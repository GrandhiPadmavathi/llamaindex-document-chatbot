import dataclasses

import pytest

from chatbot.config import DEFAULT_CONFIG
from chatbot.indexing import index_cache_key
from chatbot.llm import build_llm


def test_cache_key_depends_on_embedding_model_and_chunking():
    ollama = dataclasses.replace(DEFAULT_CONFIG, embed_provider="ollama")
    hf = dataclasses.replace(DEFAULT_CONFIG, embed_provider="huggingface")
    base = index_cache_key("abc", 600, 100, ollama)

    assert base == index_cache_key("abc", 600, 100, ollama)
    assert base != index_cache_key("abc", 600, 100, hf)
    assert base != index_cache_key("abc", 800, 100, ollama)
    assert base != index_cache_key("abc", 600, 50, ollama)


def test_identities():
    groq = dataclasses.replace(DEFAULT_CONFIG, llm_provider="groq", groq_model="llama-3.1-8b-instant")
    assert groq.llm_identity == "Groq · llama-3.1-8b-instant"
    hf = dataclasses.replace(DEFAULT_CONFIG, embed_provider="huggingface", hf_embed_model="BAAI/bge-small-en-v1.5")
    assert hf.embed_identity == "huggingface:BAAI/bge-small-en-v1.5"


def test_groq_requires_key():
    cfg = dataclasses.replace(DEFAULT_CONFIG, llm_provider="groq", groq_api_key="")
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        build_llm(cfg)


def test_groq_builds_with_key():
    cfg = dataclasses.replace(DEFAULT_CONFIG, llm_provider="groq", groq_api_key="test-key")
    llm = build_llm(cfg)
    assert llm.metadata.model_name == cfg.groq_model


@pytest.mark.slow
def test_huggingface_embeddings_load_and_embed(capsys):
    from chatbot.embeddings import configure_embed_model

    cfg = dataclasses.replace(DEFAULT_CONFIG, embed_provider="huggingface")
    try:
        # torch crashes the Windows interpreter while pytest captures output during model load
        with capsys.disabled():
            model = configure_embed_model(cfg)
            vec = model.get_text_embedding("How long do card refunds take?")
        assert len(vec) == 384
    finally:
        configure_embed_model(for_testing=True)
