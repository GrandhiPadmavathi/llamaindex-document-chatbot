from dataclasses import dataclass

from llama_index.core.schema import NodeWithScore, TextNode


@dataclass
class SourceCitation:
    index: int
    filename: str
    page_label: str | None
    snippet: str
    node_id: str | None = None

    def label(self) -> str:
        if self.page_label:
            return f"{self.filename} (page {self.page_label})"
        return self.filename


def snippet_from_text(text: str, max_len: int = 280) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def citations_from_nodes(nodes: list[NodeWithScore]) -> list[SourceCitation]:
    citations: list[SourceCitation] = []
    for i, nws in enumerate(nodes, start=1):
        node = nws.node
        if not isinstance(node, TextNode):
            text = getattr(node, "text", str(node))
            meta = getattr(node, "metadata", {}) or {}
        else:
            text = node.text
            meta = node.metadata or {}

        citations.append(
            SourceCitation(
                index=i,
                filename=str(meta.get("filename", "document")),
                page_label=meta.get("page_label"),
                snippet=snippet_from_text(text),
                node_id=getattr(node, "node_id", None),
            )
        )
    return citations


def format_context_block(citations: list[SourceCitation], nodes: list[NodeWithScore]) -> str:
    lines: list[str] = []
    for cite, nws in zip(citations, nodes):
        node = nws.node
        body = node.text if hasattr(node, "text") else str(node)
        header = f"[{cite.index}] {cite.label()}"
        lines.append(f"{header}\n{body}")
    return "\n\n".join(lines)


def format_sources_markdown(citations: list[SourceCitation]) -> str:
    parts: list[str] = []
    for c in citations:
        loc = c.label()
        parts.append(f"**[{c.index}]** {loc}\n> {c.snippet}")
    return "\n\n".join(parts)


def is_refusal_answer(text: str, refusal_phrase: str) -> bool:
    return refusal_phrase.lower() in text.lower()
