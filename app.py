import streamlit as st

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


@st.cache_resource(show_spinner=False)
def sample_documents():
    return load_documents_from_directory(config.sample_docs_dir)


def build_index(documents, file_hash: str, settings: RetrievalSettings):
    configure_llama_index(config, temperature=settings.temperature)
    return get_or_create_index(
        documents,
        file_hash,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        config=config,
        cache_manager=cache_manager,
    )


@st.cache_resource(show_spinner=False)
def sample_index(chunk_size: int, chunk_overlap: int):
    docs = sample_documents()
    settings = RetrievalSettings(chunk_size, chunk_overlap, config.default_top_k, True, False, 0.1)
    return build_index(docs, content_fingerprint(docs), settings)


with st.sidebar:
    st.title("Documents")
    source = st.radio(
        "Ask questions about",
        ["Sample documents", "Upload your own"],
        index=0 if config.demo_mode else 1,
        help="The samples are fictional support guides for a mobile operator.",
    )
    file = None
    if source == "Sample documents":
        names = sorted({d.metadata.get("filename", "?") for d in sample_documents()})
        st.markdown("\n".join(f"- `{n}`" for n in names))
    else:
        limit = f" (max {config.demo_max_upload_mb:g} MB)" if config.demo_mode else ""
        file = st.file_uploader(f"Upload a document{limit}", type=UPLOAD_TYPES)
        if config.demo_mode:
            st.caption("Uploads are processed in memory for this session only. Don't upload private data.")

    st.divider()
    st.subheader("Retrieval settings")
    chunk_size = st.slider("Chunk size", 200, 1200, config.default_chunk_size, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, config.default_chunk_overlap, 25)
    top_k = st.slider("Top K chunks", 1, 15, config.default_top_k)
    hybrid = st.toggle("Hybrid search (BM25 + vector)", value=True)
    rerank = st.toggle("Cross-encoder re-rank", value=False)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    faithfulness_check = st.toggle("Faithfulness check (LLM judge)", value=False)

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
    with st.spinner("Loading or building index..."):
        index = build_index(documents, file_hash, settings)
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
    if config.demo_mode and st.session_state.questions_asked >= config.demo_max_questions:
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
