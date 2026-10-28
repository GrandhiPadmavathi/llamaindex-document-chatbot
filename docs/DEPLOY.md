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

Spaces have no Ollama daemon, so the hosted demo uses:

| Part | Setting |
|---|---|
| LLM | `LLM_PROVIDER=groq`, `GROQ_MODEL=qwen/qwen3.8-27b` (free tier) |
| Embeddings | `EMBED_PROVIDER=huggingface`, `BAAI/bge-small-en-v1.5` on CPU (baked into the image) |
| Limits | `DEMO_MODE=1`: sample documents preloaded, 25 questions per session, 2 MB uploads, 500-character questions |

`deploy/space/` holds the Space's `Dockerfile` (non-root user, CPU-only torch, models pre-downloaded) and its
`README.md` (Space metadata). To deploy:

```bash
# PowerShell: $env:HF_TOKEN="..."; $env:GROQ_API_KEY="..."
export HF_TOKEN=...        # Hugging Face token with write access
export GROQ_API_KEY=...    # stored as an encrypted Space secret
python scripts/deploy_space.py --space <hf-username>/document-chatbot
```

The script creates the Space if needed, sets `GROQ_API_KEY` as a secret and uploads only the app code and sample
documents. The first build takes a few minutes.

Run the same configuration locally (no Ollama needed):

```bash
LLM_PROVIDER=groq EMBED_PROVIDER=huggingface DEMO_MODE=1 streamlit run app.py
```

Indexes are cached per document, embedding model and chunk settings, so switching providers never mixes vectors.

## Sign-in and roles

Streamlit's built-in OIDC support (`st.login`, needs `Authlib`) works with any OpenID Connect provider. Register a
web application with the provider, set its redirect URI to `<app url>/oauth2callback`, and put the client in
`.streamlit/secrets.toml` (git-ignored; on Hugging Face Spaces, add the same values as Space secrets):

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "a long random string"   # python -c "import secrets; print(secrets.token_urlsafe(32))"
client_id = "..."
client_secret = "..."
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
# Microsoft Entra ID: https://login.microsoftonline.com/<tenant-id>/v2.0/.well-known/openid-configuration
```

Then choose who gets in and what they can do:

```bash
AUTH_REQUIRED=1 AUTH_ALLOWED=@yourcompany.com AUTH_CONTRIBUTORS=alice@yourcompany.com \
AUTH_ADMINS=you@yourcompany.com streamlit run app.py
```

If `AUTH_REQUIRED=1` and no `[auth]` section exists, the app shows a configuration error instead of opening up.

## Environment variables

See `.env.example` for `OLLAMA_BASE_URL`, `LLM_MODEL`, `EMBED_MODEL`, and optional OpenAI-compatible settings.
