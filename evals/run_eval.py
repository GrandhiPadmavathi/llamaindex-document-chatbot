"""
Run retrieval and answer evaluation against sample_docs.

Compares: vector-only, hybrid (BM25+vector+RRF), hybrid+rerank.
Requires Ollama with LLM and embedding models for answer metrics.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chatbot.answering import generate_answer  # noqa: E402
from chatbot.config import DEFAULT_CONFIG, RetrievalSettings  # noqa: E402
from chatbot.indexing import build_index_from_documents  # noqa: E402
from chatbot.loaders import content_fingerprint, load_documents_from_directory  # noqa: E402
from chatbot.llm import configure_llama_index  # noqa: E402
from chatbot.prompts import REFUSAL_PHRASE  # noqa: E402
from chatbot.retrieval import (  # noqa: E402
    CrossEncoderReranker,
    build_retriever,
    retrieve_bm25,
    retrieve_vector,
)
from chatbot.citations import is_refusal_answer  # noqa: E402


@dataclass
class EvalConfig:
    name: str
    hybrid: bool
    rerank: bool


CONFIGS = [
    EvalConfig("vector_only", hybrid=False, rerank=False),
    EvalConfig("hybrid_rrf", hybrid=True, rerank=False),
    EvalConfig("hybrid_rrf_rerank", hybrid=True, rerank=True),
]


def load_questions(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def hit_rate_and_mrr(nodes, expected_source: str, k: int) -> tuple[float, float]:
    if not expected_source:
        return 0.0, 0.0
    filenames = []
    for nws in nodes:
        meta = getattr(nws.node, "metadata", {}) or {}
        filenames.append(str(meta.get("filename", "")))
    hit = 0.0
    rr = 0.0
    for rank, name in enumerate(filenames[:k], start=1):
        if expected_source in name or name.endswith(expected_source):
            hit = 1.0
            rr = 1.0 / rank
            break
    return hit, rr


def answer_matches(question: dict, answer: str) -> tuple[bool, bool]:
    """Returns (correct_or_refusal_ok, is_refusal)."""
    refused = is_refusal_answer(answer, REFUSAL_PHRASE)
    if question.get("unanswerable"):
        return refused, refused
    if refused:
        return False, True
    facts = question.get("key_facts") or []
    if not facts:
        return True, False
    lower = answer.lower()
    for fact in facts:
        if str(fact).lower() in lower:
            return True, False
    return False, False


def evaluate_config(eval_cfg: EvalConfig, index, questions, llm, base_settings: RetrievalSettings):
    settings = RetrievalSettings(
        chunk_size=base_settings.chunk_size,
        chunk_overlap=base_settings.chunk_overlap,
        top_k=base_settings.top_k,
        hybrid=eval_cfg.hybrid,
        rerank=eval_cfg.rerank,
        temperature=base_settings.temperature,
    )
    reranker = CrossEncoderReranker(DEFAULT_CONFIG.cross_encoder_model) if eval_cfg.rerank else None
    rerank_note = None
    if eval_cfg.rerank and reranker and not reranker.available:
        rerank_note = reranker.error or "reranker unavailable"

    retrieve = build_retriever(
        index,
        settings,
        reranker_factory=lambda: reranker or CrossEncoderReranker(DEFAULT_CONFIG.cross_encoder_model),
    )

    hits = []
    mrrs = []
    answer_ok = []
    refusals = []
    latencies: list[float] = []
    unanswerable_ok: list[bool] = []

    for q in questions:
        t0 = time.perf_counter()
        nodes = retrieve(q["question"])
        h, m = hit_rate_and_mrr(nodes, q.get("expected_source", ""), settings.top_k)
        hits.append(h)
        mrrs.append(m)

        ans = generate_answer(
            index,
            q["question"],
            llm,
            settings,
            faithfulness_check=False,
            reranker=reranker,
        )
        latencies.append(time.perf_counter() - t0)
        ok, refused = answer_matches(q, ans.text)
        answer_ok.append(ok)
        refusals.append(refused)
        if q.get("unanswerable"):
            unanswerable_ok.append(refused)

    n = len(questions)
    n_unanswerable = len(unanswerable_ok)
    return {
        "name": eval_cfg.name,
        "hit_rate_at_k": sum(hits) / n if n else 0.0,
        "mrr": sum(mrrs) / n if n else 0.0,
        "answer_accuracy": sum(answer_ok) / n if n else 0.0,
        "refusal_rate": sum(refusals) / n if n else 0.0,
        "refusal_accuracy_unanswerable": (
            sum(unanswerable_ok) / n_unanswerable if n_unanswerable else 0.0
        ),
        "avg_latency_s": sum(latencies) / n if n else 0.0,
        "rerank_available": None if not eval_cfg.rerank else (reranker.available if reranker else False),
        "rerank_note": rerank_note,
    }


def write_results_md(results: list[dict], out_path: Path, meta: dict) -> None:
    lines = [
        "# Evaluation results",
        "",
        f"Generated: {meta['timestamp']} (UTC)",
        "",
        "## Environment",
        "",
        f"- Ollama base URL: `{meta['ollama_base_url']}`",
        f"- LLM: `{meta['llm_model']}`",
        f"- Embeddings: `{meta['embed_model']}`",
        f"- Questions: {meta['num_questions']}",
        f"- Chunk size / overlap / top_k: {meta['chunk_size']} / {meta['chunk_overlap']} / {meta['top_k']}",
        "",
        "## Retrieval & answer metrics",
        "",
        "| Configuration | Hit@k | MRR | Answer accuracy | Refusal acc. (unanswerable) | Avg latency (s) | Reranker |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        rerank_col = "n/a"
        if r.get("rerank_available") is True:
            rerank_col = "yes"
        elif r.get("rerank_available") is False:
            rerank_col = f"no ({r.get('rerank_note', 'unavailable')})"
        lines.append(
            f"| {r['name']} | {r['hit_rate_at_k']:.3f} | {r['mrr']:.3f} | "
            f"{r['answer_accuracy']:.3f} | {r['refusal_accuracy_unanswerable']:.3f} | "
            f"{r['avg_latency_s']:.2f} | {rerank_col} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Hit@k and MRR use the expected source filename from `questions.jsonl`.",
            "- Answer accuracy: key-fact substring match for answerable questions; "
            f"refusal phrase for unanswerable ones (`{REFUSAL_PHRASE}`).",
            "- Metrics are computed on the machine that ran this script; do not treat them as benchmarks.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    sample_dir = ROOT / "sample_docs"
    questions_path = ROOT / "evals" / "questions.jsonl"
    out_md = ROOT / "evals" / "results.md"

    from scripts.generate_sample_pdf import main as gen_pdf

    gen_pdf()

    documents = load_documents_from_directory(sample_dir)
    if not documents:
        print("No sample documents found.")
        return 1

    _ = content_fingerprint(documents)
    settings = RetrievalSettings(
        chunk_size=DEFAULT_CONFIG.default_chunk_size,
        chunk_overlap=DEFAULT_CONFIG.default_chunk_overlap,
        top_k=DEFAULT_CONFIG.default_top_k,
        hybrid=True,
        rerank=False,
        temperature=0.1,
    )

    print("Configuring Ollama LLM and embeddings...")
    llm = configure_llama_index(DEFAULT_CONFIG, temperature=settings.temperature)
    print("Building index...")
    index = build_index_from_documents(
        documents, settings.chunk_size, settings.chunk_overlap
    )

    questions = load_questions(questions_path)
    meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "ollama_base_url": DEFAULT_CONFIG.ollama_base_url,
        "llm_model": DEFAULT_CONFIG.llm_model,
        "embed_model": DEFAULT_CONFIG.embed_model,
        "num_questions": len(questions),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
    }
    results_dir = ROOT / "evals" / "results"
    results_dir.mkdir(exist_ok=True)
    rerun = "--rerun" in sys.argv
    results = []
    for cfg in CONFIGS:
        cfg_path = results_dir / f"{cfg.name}.json"
        if cfg_path.exists() and not rerun:
            print(f"Skipping {cfg.name} (saved in {cfg_path.name}; use --rerun to redo)", flush=True)
            results.append(json.loads(cfg_path.read_text(encoding="utf-8")))
        else:
            print(f"Evaluating {cfg.name}...", flush=True)
            result = evaluate_config(cfg, index, questions, llm, settings)
            result["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            cfg_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            results.append(result)
            print(result, flush=True)
        write_results_md(results, out_md, meta)
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
