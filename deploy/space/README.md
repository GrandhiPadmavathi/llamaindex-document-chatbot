---
title: Document Chatbot (RAG)
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: RAG chatbot with hybrid search, re-ranking and citations
---

# Document Chatbot — live demo

Ask questions about a set of fictional mobile-operator support guides, or upload your own small document.

- Hybrid retrieval: BM25 + vector search fused with Reciprocal Rank Fusion, optional cross-encoder re-ranking
- Answers cite the exact chunks they used and refuse when the documents don't contain the answer
- Model: Llama 3.1 8B via Groq; embeddings: `BAAI/bge-small-en-v1.5` on CPU

Source code, evaluation results and the local (Ollama) setup:
https://github.com/GrandhiPadmavathi/llamaindex-document-chatbot

Demo limits: 25 questions per session, uploads up to 2 MB, processed in memory only. Please don't upload private data.
