"""Optional live Ollama smoke test — skipped when Ollama is unreachable."""

import urllib.error
import urllib.request

import pytest

from chatbot.config import DEFAULT_CONFIG


def _ollama_reachable() -> bool:
    url = f"{DEFAULT_CONFIG.ollama_base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


@pytest.mark.ollama
@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not reachable at OLLAMA_BASE_URL")
def test_ollama_tags_endpoint():
    url = f"{DEFAULT_CONFIG.ollama_base_url.rstrip('/')}/api/tags"
    with urllib.request.urlopen(url, timeout=5) as resp:
        assert resp.status == 200
