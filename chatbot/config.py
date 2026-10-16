import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "mistral")
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_api_base: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    cache_dir: str = os.getenv("LLAMINDEX_CACHE_DIR", "llamaindex_cache")
    bot_cache_dir: str = os.getenv("BOT_CACHE_DIR", "bot_cache")
    default_chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    default_chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    default_top_k: int = int(os.getenv("TOP_K", "5"))
    cross_encoder_model: str = os.getenv(
        "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


DEFAULT_CONFIG = AppConfig()


@dataclass
class RetrievalSettings:
    chunk_size: int
    chunk_overlap: int
    top_k: int
    hybrid: bool
    rerank: bool
    temperature: float
