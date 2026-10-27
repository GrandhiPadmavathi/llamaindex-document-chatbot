import streamlit as st

from chatbot import access
from chatbot.access import LOCAL_USER, AccessPolicy, Principal, QuotaExceeded, UsageLedger
from chatbot.answering import generate_answer
from chatbot.cache import BotCacheManager
from chatbot.citations import format_sources_markdown
from chatbot.config import DEFAULT_CONFIG, RetrievalSettings
from chatbot.indexing import get_or_create_index
from chatbot.llm import configure_llama_index
from chatbot.loaders import (
    content_fingerprint,
    extract_documents_from_upload,
    load_documents_from_directory,
    read_upload_as_bytes,
)

config = DEFAULT_CONFIG
cache_manager = BotCacheManager(cache_dir=config.bot_cache_dir)

UPLOAD_TYPES = ["pdf", "docx", "txt", "md", "csv", "xlsx", "json", "html"]
if not config.demo_mode:
    UPLOAD_TYPES += ["xls", "rtf", "py", "js", "xml", "log"]

SAMPLE_QUESTIONS = [
    "How long do card refunds take once approved?",
    "When does an eSIM QR code expire?",
    "What is the maximum data rollover amount for eligible plans?",
    "What is OrbitMobile's corporate headquarters street address?",
]

st.set_page_config(page_title="Document Chatbot", page_icon="📚", layout="wide")
st.header("Document Chatbot")
st.caption(
    f"RAG with LlamaIndex · hybrid BM25 + vector search · RRF · optional cross-encoder re-rank · "
    f"citations · model: **{config.llm_identity}**"
)

policy = AccessPolicy.from_env()
persist_uploads = not (policy.required or config.demo_mode)


@st.cache_resource(show_spinner=False)
def usage_ledger() -> UsageLedger:
    return UsageLedger()


def signed_in_user() -> Principal:
    if not policy.required:
        return LOCAL_USER
    if not access.auth_configured():
        st.error(
            "Sign-in is required (`AUTH_REQUIRED=1`) but no identity provider is configured. "
            "Add an `[auth]` section to `.streamlit/secrets.toml`; see docs/DEPLOY.md."
        )
        st.stop()
    identity = access.read_identity()
    if identity is None:
        st.info("Sign in to ask questions about the documents.")
        st.button("Sign in", type="primary", on_click=st.login, args=(policy.provider,) if policy.provider else ())
        st.stop()
    user = policy.principal(identity)
    if user is None:
        email = identity.get("email") or identity.get("preferred_username") or "This account"
        if st.session_state.get("denied") != email:
            st.session_state.denied = email
            access.log_event("access_denied", email=email, reason="account not allowed or email not verified")
        st.error(f"{email} is not allowed to use this app. Ask an administrator for access.")
        st.button("Sign out", on_click=st.logout)
        st.stop()
    if st.session_state.get("signed_in_as") != user.email:
        st.session_state.signed_in_as = user.email
        access.log_event("login", user=user.email, role=user.role)
    return user


user = signed_in_user()


@st.cache_resource(show_spinner=False)
def sample_documents():
    return load_documents_from_directory(config.sample_docs_dir)


def build_index(documents, file_hash: str, settings: RetrievalSettings, persist: bool = True):
    configure_llama_index(config, temperature=settings.temperature)
    return get_or_create_index(
        documents,
        file_hash,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        config=config,
        cache_manager=cache_manager,
        persist=persist,
    )


@st.cache_resource(show_spinner=False)
def sample_index(chunk_size: int, chunk_overlap: int):
    docs = sample_documents()
    settings = RetrievalSettings(chunk_size, chunk_overlap, config.default_top_k, True, False, 0.1)
    return build_index(docs, content_fingerprint(docs), settings)


with st.sidebar:
    if policy.required:
        st.caption(f"Signed in as **{user.name}** · {user.role}")
        st.button("Sign out", on_click=st.logout)
        if user.can("usage"):
            with st.expander("Usage today"):
                rows = usage_ledger().today()
                if rows:
                    st.dataframe(rows, hide_index=True)
                else:
                    st.caption("No activity yet.")
        st.divider()
    st.title("Documents")
    sources = ["Sample documents", "Upload your own"] if user.can("upload") else ["Sample documents"]
    source = st.radio(
        "Ask questions about",
        sources,
        index=0 if config.demo_mode or len(sources) == 1 else 1,
        help="The samples are fictional support guides for a mobile operator.",
    )
    if not user.can("upload"):
        st.caption("Uploading your own documents needs contributor access.")
    file = None
    if source == "Sample documents":
        names = sorted({d.metadata.get("filename", "?") for d in sample_documents()})
        st.markdown("\n".join(f"- `{n}`" for n in names))
    else:
        limit = f" (max {config.demo_max_upload_mb:g} MB)" if config.demo_mode else ""
        file = st.file_uploader(f"Upload a document{limit}", type=UPLOAD_TYPES)
        if not persist_uploads:
            st.caption("Uploads are indexed in memory for this session only and never written to disk.")

    st.divider()
    st.subheader("Retrieval settings")
    chunk_size = st.slider("Chunk size", 200, 1200, config.default_chunk_size, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, config.default_chunk_overlap, 25)
    top_k = st.slider("Top K chunks", 1, 15, config.default_top_k)
    hybrid = st.toggle("Hybrid search (BM25 + vector)", value=True)
    locked = None if user.can("advanced") else "Needs contributor access (extra model calls)."
    rerank = st.toggle("Cross-encoder re-rank", value=False, disabled=bool(locked), help=locked)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    faithfulness_check = st.toggle("Faithfulness check (LLM judge)", value=False, disabled=bool(locked), help=locked)

    if st.button("Clear chat"):
        st.session_state.pop("messages", None)

settings = RetrievalSettings(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    top_k=top_k,
    hybrid=hybrid,
    rerank=rerank,
    temperature=temperature,
)

if source == "Sample documents":
    chat_key = "samples"
    with st.spinner("Indexing the sample documents (first load only)..."):
        index = sample_index(chunk_size, chunk_overlap)
    llm = configure_llama_index(config, temperature=temperature)
else:
    if file is None:
        st.markdown(
            "Upload a PDF, DOCX, TXT or other supported file in the sidebar, or switch to **Sample documents**. "
            "Answers include inline citations **[1]**, **[2]** and expandable source snippets."
        )
        st.stop()
    if config.demo_mode and file.size > config.demo_max_upload_mb * 1024 * 1024:
        st.error(f"This demo accepts files up to {config.demo_max_upload_mb:g} MB.")
        st.stop()
    with st.spinner(f"Processing {file.name}..."):
        buffer = read_upload_as_bytes(file)
        buffer.name = file.name  # type: ignore[attr-defined]
        documents = extract_documents_from_upload(buffer, filename=file.name)
    if not documents:
        st.error("No text content found in the file.")
        st.stop()
    file_hash = content_fingerprint(documents)
    chat_key = f"upload:{file_hash}"
    uploaded = st.session_state.setdefault("uploaded", set())
    if policy.required and file_hash not in uploaded:
        try:
            usage_ledger().consume(user.email, "upload", policy.uploads_per_day)
        except QuotaExceeded as exc:
            access.log_event("quota_exceeded", user=user.email, kind="upload")
            st.warning(str(exc))
            st.stop()
        uploaded.add(file_hash)
        access.log_event("upload", user=user.email, file=file.name, bytes=file.size, sha=file_hash[:12])
    index_key = (file_hash, chunk_size, chunk_overlap)
    session_index = st.session_state.get("session_index")
    if not persist_uploads and session_index and session_index[0] == index_key:
        index = session_index[1]
    else:
        with st.spinner("Loading or building index..."):
            index = build_index(documents, file_hash, settings, persist=persist_uploads)
        if not persist_uploads:
            st.session_state.session_index = (index_key, index)
    llm = configure_llama_index(config, temperature=temperature)

if st.session_state.get("chat_key") != chat_key:
    st.session_state.messages = []
    st.session_state.chat_key = chat_key
st.session_state.setdefault("messages", [])
st.session_state.setdefault("questions_asked", 0)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                st.markdown(message["sources"])

prompt = st.chat_input("Ask a question about the documents")
if source == "Sample documents" and not st.session_state.messages and not prompt:
    st.markdown("**Try one of these** (the last one isn't in the documents, so the bot should refuse):")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, question in zip(cols, SAMPLE_QUESTIONS):
        if col.button(question, use_container_width=True):
            prompt = question

if prompt:
    if policy.required:
        try:
            usage_ledger().consume(user.email, "question", policy.questions_per_day)
        except QuotaExceeded as exc:
            access.log_event("quota_exceeded", user=user.email, kind="question")
            st.warning(str(exc))
            st.stop()
    elif config.demo_mode and st.session_state.questions_asked >= config.demo_max_questions:
        st.warning(f"This public demo allows {config.demo_max_questions} questions per session. Refresh to start over.")
        st.stop()
    prompt = prompt.strip()[: config.max_question_chars]
    st.session_state.questions_asked += 1
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            try:
                result = generate_answer(
                    index, prompt, llm, settings, config=config, faithfulness_check=faithfulness_check
                )
            except Exception as exc:  # surface provider errors (rate limits, auth) instead of a stack trace
                st.error(f"The model call failed: {type(exc).__name__}. Please try again in a moment.")
                st.stop()

        if result.faithfulness == "supported":
            st.success("Faithfulness: supported by context")
        elif result.faithfulness == "unsupported":
            st.warning("Faithfulness: not fully supported by context")
        elif result.refused:
            st.info("Answer grounded refusal")

        st.markdown(result.text)
        sources_md = format_sources_markdown(result.citations)
        with st.expander(f"Sources ({len(result.citations)})"):
            st.markdown(sources_md)

    st.session_state.messages.append({"role": "assistant", "content": result.text, "sources": sources_md})
    if policy.required:
        access.log_event("question", user=user.email, source=chat_key.split(":")[0], chars=len(prompt),
                         refused=result.refused, citations=len(result.citations))
