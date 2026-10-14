# Local Document Chatbot (LlamaIndex + Ollama)

Upload a document and ask questions about it. Answers are generated with
Retrieval-Augmented Generation (RAG) using models that run entirely on your
own machine through [Ollama](https://ollama.com) - no API keys, no data leaves
your computer.

## How it works

1. **Upload** a file in the sidebar.
2. **Extract** the text (PyPDF2, python-docx, pandas, ...).
3. **Split** the text into ~600-character chunks with 100 characters overlap.
4. **Embed** each chunk with `nomic-embed-text` and store it in a LlamaIndex vector index.
5. **Retrieve** the 5 chunks most similar to your question.
6. **Answer** with `mistral`, using only the retrieved chunks as context.

Each document's index is saved to `llamaindex_cache/` (keyed by a hash of its
text), so uploading the same document again loads instantly.

## Supported file types

pdf, docx, txt, csv, xlsx, xls, json, md, rtf, py, js, html, xml, log

## Setup

1. Install Ollama and pull the two models:

   ```bash
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

2. Install the Python dependencies (Python 3.10+):

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

   Then open http://localhost:8501 in your browser.

## Project structure

```
app.py             Streamlit UI, text extraction, indexing and querying
cache_manager.py   Small pickle-backed cache that remembers saved index locations
requirements.txt   Python dependencies
```

## Tech stack

Streamlit · LlamaIndex · Ollama (mistral, nomic-embed-text)
