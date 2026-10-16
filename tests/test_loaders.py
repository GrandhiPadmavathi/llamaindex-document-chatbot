from io import BytesIO

from chatbot.loaders import documents_from_text, extract_documents_from_upload


def test_documents_from_text_metadata():
    docs = documents_from_text("Hello world", "demo.txt")
    assert len(docs) == 1
    assert docs[0].metadata["filename"] == "demo.txt"


def test_extract_markdown_upload():
    data = BytesIO(b"# Title\n\nBody text")
    data.name = "note.md"  # type: ignore[attr-defined]
    docs = extract_documents_from_upload(data, filename="note.md")
    assert "Body text" in docs[0].text
