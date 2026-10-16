"""Pytest fixtures — unit tests never touch Ollama or OpenAI embeddings."""

import pytest

from chatbot.embeddings import configure_embed_model, mock_embed_model


@pytest.fixture(scope="session", autouse=True)
def _embed_model_for_tests():
    configure_embed_model(for_testing=True)
    yield


@pytest.fixture
def embed_model():
    return mock_embed_model()
