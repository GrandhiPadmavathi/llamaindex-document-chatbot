import streamlit as st

from chatbot.answering import generate_answer
from chatbot.cache import BotCacheManager
from chatbot.config import DEFAULT_CONFIG, RetrievalSettings
from chatbot.loaders import content_fingerprint, extract_documents_from_upload, read_upload_as_bytes
from chatbot.indexing import get_or_create_index
from chatbot.llm import configure_llama_index
from chatbot.citations import format_sources_markdown

st.set_page_config(page_title="Document Chatbot", page_icon="📚", layout="wide")
st.header("Document Chatbot (LlamaIndex + Ollama)")

config = DEFAULT_CONFIG
cache_manager = BotCacheManager(cache_dir=config.bot_cache_dir)

with st.sidebar:
    st.title("Your documents")
    file = st.file_uploader(
        "Upload a document and ask questions",
        type=[
            "pdf",
            "docx",
            "txt",
            "csv",
            "xlsx",
            "xls",
            "json",
            "md",
            "rtf",
            "py",
            "js",
            "html",
            "xml",
            "log",
        ],
    )

    st.divider()
    st.subheader("Retrieval settings")
    chunk_size = st.slider("Chunk size", 200, 1200, config.default_chunk_size, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, config.default_chunk_overlap, 25)
    top_k = st.slider("Top K chunks", 1, 15, config.default_top_k)
    hybrid = st.toggle("Hybrid search (BM25 + vector)", value=True)
    rerank = st.toggle("Cross-encoder re-rank", value=False)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    faithfulness_check = st.toggle("Faithfulness check (LLM judge)", value=False)

    if file is not None:
        st.info(f"File: **{file.name}** ({file.size / 1024:.1f} KB)")

if file is not None:
    settings = RetrievalSettings(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        hybrid=hybrid,
        rerank=rerank,
        temperature=temperature,
    )

    with st.spinner(f"Processing {file.name}..."):
        buffer = read_upload_as_bytes(file)
        buffer.name = file.name  # type: ignore[attr-defined]
        documents = extract_documents_from_upload(buffer, filename=file.name)

    if not documents:
        st.error("No text content found in the file.")
        st.stop()

    file_hash = content_fingerprint(documents)

    with st.spinner("Loading or building index..."):
        llm = configure_llama_index(config, temperature=temperature)
        index = get_or_create_index(
            documents,
            file_hash,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            config=config,
            cache_manager=cache_manager,
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    st.markdown(message["sources"])

    if prompt := st.chat_input("Ask a question about your document"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                result = generate_answer(
                    index,
                    prompt,
                    llm,
                    settings,
                    config=config,
                    faithfulness_check=faithfulness_check,
                )

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

            st.session_state.messages.append(
                {"role": "assistant", "content": result.text, "sources": sources_md}
            )
else:
    st.markdown(
        "Upload a PDF, DOCX, TXT, or other supported file in the sidebar to begin. "
        "Answers include inline citations **[1]**, **[2]** and expandable source snippets."
    )
