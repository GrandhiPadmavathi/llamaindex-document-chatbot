# Evaluation results

Generated: 2026-10-16 06:02:22 (UTC)

## Environment

- Ollama base URL: `http://localhost:11434`
- LLM: `mistral`
- Embeddings: `nomic-embed-text`
- Questions: 26
- Chunk size / overlap / top_k: 600 / 100 / 5

## Retrieval & answer metrics

| Configuration | Hit@k | MRR | Answer accuracy | Refusal acc. (unanswerable) | Avg latency (s) | Reranker |
|---|---:|---:|---:|---:|---:|---|
| vector_only | 0.923 | 0.878 | 1.000 | 1.000 | 162.56 | n/a |
| hybrid_rrf | 0.923 | 0.865 | 1.000 | 1.000 | 97.37 | n/a |
| hybrid_rrf_rerank | 0.923 | 0.923 | 0.962 | 0.500 | 98.29 | yes |

## Notes

- Hit@k and MRR use the expected source filename from `questions.jsonl`.
- Answer accuracy: key-fact substring match for answerable questions; refusal phrase for unanswerable ones (`I could not find this in the documents`).
- Metrics are computed on the machine that ran this script; do not treat them as benchmarks.
