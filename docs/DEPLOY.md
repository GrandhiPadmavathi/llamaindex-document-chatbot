# Deployment guide

## Docker (local)

The Streamlit app runs in the container; **Ollama stays on the host** by default.

1. Install [Ollama](https://ollama.com) on your machine and pull models:

   ```bash
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

2. Build and run:

   ```bash
   docker build -t orbit-doc-chatbot .
   docker run --rm -p 8501:8501 \
     -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
     orbit-doc-chatbot
   ```

   On Linux without `host.docker.internal`, use `--add-host=host.docker.internal:host-gateway` or point `OLLAMA_BASE_URL` at your host LAN IP.

3. Open http://localhost:8501

Persisted indexes and caches are written inside the container unless you mount volumes for `llamaindex_cache/` and `bot_cache/`.

## Hugging Face Spaces

Spaces run the Docker/Streamlit app in Hugging Face infrastructure. They **do not** include a local Ollama daemon.

Options:

| Approach | Notes |
|----------|--------|
| Remote Ollama | Run Ollama on a VM you control; set `OLLAMA_BASE_URL` to that HTTPS/HTTP endpoint (lock down firewall + TLS). |
| OpenAI-compatible API | Set `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and optionally `OPENAI_API_BASE` for Azure, Groq, Together, etc. Embeddings still use Ollama unless you extend the code. |
| CPU embedding swap | For a full cloud setup, replace `OllamaEmbedding` with a hosted embedding API (not enabled by default in this repo). |

Copy `.env.example` to `.env` locally; on Spaces, use **Repository secrets** / Space variables for the same keys.

## Environment variables

See `.env.example` for `OLLAMA_BASE_URL`, `LLM_MODEL`, `EMBED_MODEL`, and optional OpenAI-compatible settings.
