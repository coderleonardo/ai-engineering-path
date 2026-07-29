# Running This Module

A minimal walkthrough for running the three reference scripts in this folder end-to-end: index a folder
of documents, serve grounded/cited answers over an API, and query them from a web UI. See
[`rag.md`](./rag.md) for how the pipeline is designed and [`llmops.md`](./llmops.md) for operating it
beyond a local run.

## Prerequisites

- Python 3.11+
- Docker (used to run Qdrant, the vector database)
- An API key for any OpenAI-compatible LLM provider (OpenAI itself, or another provider exposing the
  same API shape)

## 1. Install dependencies

These scripts use packages outside this repo's shared `pyproject.toml` (see each script's own docstring
for exactly which ones it needs). In a virtual environment of your choice:

```bash
pip install pypdf python-docx python-pptx langchain-text-splitters langchain-huggingface \
    sentence-transformers langchain-qdrant qdrant-client python-dotenv fastapi uvicorn openai \
    pydantic streamlit requests
```

## 2. Start Qdrant with Docker

Qdrant (the vector database `document_indexer.py` and `rag_api.py` both talk to) runs as a container,
not a Python dependency:

```bash
docker run --name qdrant -d -p 6333:6333 qdrant/qdrant
```

- `-d` runs it detached (in the background).
- `-p 6333:6333` publishes Qdrant's REST API on `localhost:6333` — this is what `QDRANT_URL` (default
  `http://localhost:6333`) points at in both scripts.
- Confirm it's up via the dashboard: http://localhost:6333/dashboard

To stop it later: `docker stop qdrant` (add `docker rm qdrant` to remove the container entirely; its
data lives inside the container's writable layer, so removing it discards the indexed collection too).

## 3. Configure environment variables

Create a `.env` file in this folder (loaded automatically by `rag_api.py` via `python-dotenv`):

```
LLM_API_KEY=your-api-key-here
# Optional -- omit to target OpenAI's own endpoint; set to point at any other
# OpenAI-compatible provider (e.g. a self-hosted server, or another provider's API).
LLM_BASE_URL=
# Optional -- defaults to gpt-4o-mini if unset.
LLM_MODEL=
# Optional -- defaults to http://localhost:6333 if unset.
QDRANT_URL=
```

## 4. Index your documents

```bash
python document_indexer.py /path/to/your/documents
```

Point this at any folder — it recurses into subdirectories and indexes every `.pdf`, `.txt`, `.docx`, and
`.pptx` file it finds, skipping anything else. Re-run this whenever the source documents change; each run
fully rebuilds the collection (see `document_indexer.py`'s comments for why).

## 5. Start the API

```bash
python rag_api.py
```

Serves on `http://0.0.0.0:8000`. Sanity-check it's running: `curl http://localhost:8000/`.

## 6. Start the web app

In a separate terminal:

```bash
streamlit run rag_streamlit_app.py
```

Opens a browser tab with a text input. Ask a question about whatever you indexed in step 4 — the answer
will cite which source chunks it used, and each cited source is shown below with a download button for
the original file.

If the API isn't running on the default address, point the app at it first:
`RAG_API_URL=http://host:port/query streamlit run rag_streamlit_app.py`.

## Shutting down

Stop the Streamlit app and the API with `Ctrl+C` in their respective terminals, then stop Qdrant:

```bash
docker stop qdrant
```
