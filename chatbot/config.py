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
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    embed_provider: str = os.getenv("EMBED_PROVIDER", "ollama").lower()
    hf_embed_model: str = os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    demo_mode: bool = os.getenv("DEMO_MODE", "0").lower() in ("1", "true", "yes")
    demo_max_questions: int = int(os.getenv("DEMO_MAX_QUESTIONS", "25"))
    demo_max_upload_mb: float = float(os.getenv("DEMO_MAX_UPLOAD_MB", "2"))
    max_question_chars: int = int(os.getenv("MAX_QUESTION_CHARS", "500"))
    sample_docs_dir: str = os.getenv("SAMPLE_DOCS_DIR", "sample_docs")

    @property
    def embed_identity(self) -> str:
        if self.embed_provider == "huggingface":
            return f"huggingface:{self.hf_embed_model}"
        return f"ollama:{self.embed_model}"

    @property
    def llm_identity(self) -> str:
        if self.llm_provider == "groq":
            return f"Groq · {self.groq_model}"
        if self.llm_provider == "openai":
            return f"OpenAI-compatible · {self.openai_model}"
        return f"Ollama · {self.llm_model}"


DEFAULT_CONFIG = AppConfig()


@dataclass
class RetrievalSettings:
    chunk_size: int
    chunk_overlap: int
    top_k: int
    hybrid: bool
    rerank: bool
    temperature: float
