from __future__ import annotations

from typing import Callable

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode

from chatbot.config import RetrievalSettings


def node_key(node: TextNode) -> str:
    return node.node_id or node.hash or node.text[:80]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked ID lists using reciprocal rank fusion (RRF)."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _nodes_by_id(index: VectorStoreIndex) -> dict[str, TextNode]:
    mapping: dict[str, TextNode] = {}
    for node in index.docstore.docs.values():
        if isinstance(node, TextNode):
            mapping[node_key(node)] = node
    return mapping


def retrieve_vector(index: VectorStoreIndex, query: str, top_k: int) -> list[NodeWithScore]:
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(query)


def retrieve_bm25(index: VectorStoreIndex, query: str, top_k: int) -> list[NodeWithScore]:
    from llama_index.retrievers.bm25 import BM25Retriever

    n_nodes = len(index.docstore.docs)
    effective_k = min(top_k, n_nodes) if n_nodes else top_k
    bm25 = BM25Retriever.from_defaults(
        docstore=index.docstore,
        similarity_top_k=effective_k,
    )
    return bm25.retrieve(query)


def fuse_retrievals(
    vector_nodes: list[NodeWithScore],
    bm25_nodes: list[NodeWithScore],
    top_k: int,
) -> list[NodeWithScore]:
    id_to_nws: dict[str, NodeWithScore] = {}
    for nws in vector_nodes + bm25_nodes:
        node = nws.node
        if not isinstance(node, TextNode):
            continue
        id_to_nws[node_key(node)] = nws

    fused_ids = reciprocal_rank_fusion(
        [
            [node_key(n.node) for n in vector_nodes if isinstance(n.node, TextNode)],
            [node_key(n.node) for n in bm25_nodes if isinstance(n.node, TextNode)],
        ]
    )
    fused: list[NodeWithScore] = []
    for item_id, score in fused_ids[:top_k]:
        if item_id in id_to_nws:
            nws = id_to_nws[item_id]
            fused.append(NodeWithScore(node=nws.node, score=score))
    return fused


class CrossEncoderReranker:
    """Optional cross-encoder re-ranker; fails gracefully if model unavailable."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self.available = False
        self.error: str | None = None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)
            self.available = True
        except Exception as exc:  # noqa: BLE001 — intentional graceful fallback
            self.error = str(exc)

    def rerank(self, query: str, nodes: list[NodeWithScore], top_k: int) -> list[NodeWithScore]:
        if not self.available or not self._model or not nodes:
            return nodes[:top_k]

        pairs = [(query, n.node.text if hasattr(n.node, "text") else str(n.node)) for n in nodes]
        scores = self._model.predict(pairs)
        scored = list(zip(nodes, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        return [
            NodeWithScore(node=nws.node, score=float(score)) for nws, score in scored[:top_k]
        ]


def build_retriever(
    index: VectorStoreIndex,
    settings: RetrievalSettings,
    reranker_factory: Callable[[], CrossEncoderReranker] | None = None,
):
    """Return a callable query -> list[NodeWithScore]."""

    def retrieve(query: str) -> list[NodeWithScore]:
        if settings.hybrid:
            vector_nodes = retrieve_vector(index, query, settings.top_k)
            bm25_nodes = retrieve_bm25(index, query, settings.top_k)
            nodes = fuse_retrievals(vector_nodes, bm25_nodes, settings.top_k)
        else:
            nodes = retrieve_vector(index, query, settings.top_k)

        if settings.rerank and reranker_factory is not None:
            reranker = reranker_factory()
            if reranker.available:
                nodes = reranker.rerank(query, nodes, settings.top_k)
        return nodes

    return retrieve
