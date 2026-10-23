from llama_index.core import Settings
from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama

from chatbot.config import AppConfig, DEFAULT_CONFIG
from chatbot.embeddings import configure_embed_model


def configure_llama_index(config: AppConfig | None = None, temperature: float = 0.1) -> LLM:
    config = config or DEFAULT_CONFIG
    llm = build_llm(config, temperature=temperature)
    Settings.llm = llm
    configure_embed_model(config)
    return llm


def build_llm(config: AppConfig | None = None, temperature: float = 0.1) -> LLM:
    config = config or DEFAULT_CONFIG
    if config.llm_provider == "groq":
        if not config.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        from llama_index.llms.groq import Groq

        return Groq(model=config.groq_model, api_key=config.groq_api_key, temperature=temperature)

    if config.llm_provider == "openai":
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        from llama_index.llms.openai import OpenAI

        return OpenAI(
            model=config.openai_model,
            api_key=config.openai_api_key,
            api_base=config.openai_api_base,
            temperature=temperature,
        )

    return Ollama(
        model=config.llm_model,
        base_url=config.ollama_base_url,
        request_timeout=1200.0,
        temperature=temperature,
    )
