"""RAG Query API — Reference Script.

Reference script. See `document_indexer.py` for building the Qdrant collection this queries, and
`rag_streamlit_app.py` for a client that consumes this API. See `rag.md` for the citation/grounding
prompting pattern used below, presented generically.

Methods covered:
- Embedding an incoming query and running `similarity_search` against Qdrant
- Building a numbered, citable context block from the retrieved chunks
- Calling an LLM through any OpenAI-compatible endpoint (OpenAI itself, or any provider exposing the
  same API shape) via `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL` environment variables
- Returning a structured response (`answer` + `sources`) so a client can resolve citations back to
  source documents without re-parsing the vector store itself

Use this as a reference when: you need a minimal RAG backend that grounds answers in retrieved
documents and cites which ones it used.

Don't use this as a reference for: the ingestion side (see `document_indexer.py`), multi-turn
conversational memory (this is single-turn: one query in, one grounded answer out), or a specific LLM
provider's SDK quirks — this deliberately targets the lowest common denominator (the OpenAI-compatible
chat completions shape) so it's portable across providers.

Requires (not part of the repo's shared pyproject.toml — install separately): fastapi, uvicorn, openai,
pydantic, python-dotenv, langchain-huggingface, sentence-transformers, qdrant-client, langchain-qdrant.

Run: python rag_api.py (serves on http://0.0.0.0:8000)
Requires a Qdrant instance populated by document_indexer.py, and a .env with at least LLM_API_KEY set.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "sentence-transformers/msmarco-bert-base-dot-v5"

# LLM_BASE_URL is intentionally optional: leaving it unset targets OpenAI's own endpoint; setting it
# points the same OpenAI SDK call at any other OpenAI-compatible provider (e.g. a self-hosted vLLM
# server, or a third-party inference API) without changing any code below.
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

if LLM_API_KEY:
    llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
else:
    llm_client = None
    print("LLM_API_KEY not set -- /query will return retrieved context without a generated answer.")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
client = QdrantClient(QDRANT_URL)
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

app = FastAPI()


class Query(BaseModel):
    query: str


@app.get("/")
async def root():
    return {"message": "RAG query API"}


@app.post("/query")
async def query(request: Query):
    results = vector_store.similarity_search(request.query, k=10)

    # Numbering each retrieved chunk and asking the model to cite that number back (see the system
    # prompt below) is what makes it possible to resolve "which source backs this claim" after the
    # fact -- the alternative (asking for a free-text citation) is far less reliable to parse.
    context = ""
    sources = []
    for i, result in enumerate(results):
        context += f"{i}\n{result.page_content}\n\n"
        sources.append({"id": i, "path": result.metadata.get("path"), "content": result.page_content})

    system_message = {
        "role": "system",
        "content": (
            "Answer the user's question using the documents provided in the context. The context "
            "contains documents that should contain an answer. Always reference the ID of the document "
            "(in brackets, e.g. [0], [1]) used to answer the question. Use as many citations and "
            "documents as necessary to answer the question."
        ),
    }
    messages = [system_message, {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {request.query}"}]

    if llm_client is None:
        return {"sources": sources, "answer": None}

    completion = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=False,
    )
    answer = completion.choices[0].message.content

    return {"sources": sources, "answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=False, workers=3)
