"""Ingest sample_docs and run one grounded question with citations (CLI smoke test)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chatbot.answering import generate_answer
from chatbot.config import DEFAULT_CONFIG, RetrievalSettings
from chatbot.indexing import build_index_from_documents
from chatbot.loaders import load_documents_from_directory
from chatbot.llm import configure_llama_index
from chatbot.citations import format_sources_markdown


def main() -> int:
    docs = load_documents_from_directory(ROOT / "sample_docs")
    settings = RetrievalSettings(
        chunk_size=600,
        chunk_overlap=100,
        top_k=5,
        hybrid=True,
        rerank=False,
        temperature=0.1,
    )
    llm = configure_llama_index(DEFAULT_CONFIG, temperature=0.1)
    index = build_index_from_documents(docs, settings.chunk_size, settings.chunk_overlap)
    question = "How much does Orbit Lite cost per month?"
    result = generate_answer(index, question, llm, settings)
    print("Q:", question)
    print("A:", result.text)
    print("Sources:")
    print(format_sources_markdown(result.citations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
