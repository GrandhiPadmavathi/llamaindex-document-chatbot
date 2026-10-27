from llama_index.core.schema import NodeWithScore, TextNode

from chatbot.citations import citations_from_nodes, format_sources_markdown


def test_sources_markdown_escapes_document_formatting():
    node = TextNode(text="## Plan overview | Plan | $15 | $25 |", metadata={"filename": "plans_faq.md"})
    md = format_sources_markdown(citations_from_nodes([NodeWithScore(node=node, score=1.0)]))
    assert "\\#\\# Plan overview \\| Plan" in md
    assert "\\$15 \\| \\$25" in md
    assert "plans\\_faq.md" in md


def test_citation_page_label():
    node = TextNode(text="Sample chunk text.", metadata={"filename": "doc.pdf", "page_label": "3"})
    cites = citations_from_nodes([NodeWithScore(node=node, score=1.0)])
    assert cites[0].label() == "doc.pdf (page 3)"
    md = format_sources_markdown(cites)
    assert "[1]" in md
    assert "Sample chunk" in md
