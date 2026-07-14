# Chroma — Vector Database

Chroma is an open-source vector database: it stores embeddings alongside their source documents and
metadata, and exposes similarity search over those embeddings at query time. It's the concrete storage/
retrieval engine behind the generic "vector database" concept covered in
[module 05](../05-prompt-engineering/vector_databases.md) — that file explains *why* a vector store enables
retrieval; this one covers what Chroma specifically provides.

## What Chroma Provides

- **Document storage** — documents and their metadata are stored together, not just raw vectors, so a
  retrieved result carries its original text and any associated fields.
- **Embeddings** — works with any embedding function (OpenAI, Cohere, Hugging Face,
  sentence-transformers, and others); Chroma stores whatever vectors it's given rather than tying you to
  one embedding provider.
- **Vector search** — similarity search over dense vectors (and sparse/hybrid search), returning the
  nearest stored documents to a query embedding.
- **Metadata filtering** — query results can be narrowed by metadata conditions, combining exact filters
  with similarity search rather than relying on similarity alone.
- **Full-text and regex search** — keyword-based search over stored documents without going through
  embeddings at all, useful when a query is better served by exact/lexical matching than semantic
  similarity.
- **Multi-modal retrieval** — the same storage/search model extends to images and audio, not just text.

## Role in a RAG Pipeline

Chroma implements the "store" and "similarity search" halves of the RAG pattern: documents are embedded
once and added to a Chroma collection, and at query time the incoming question is embedded with the same
model and used to retrieve the nearest stored documents. Wrapping a collection with `.as_retriever(...)`
turns it into the retriever component a chain (or agent) calls — the `k` passed via `search_kwargs`
controls how many nearest documents come back per query.

Reference: https://docs.trychroma.com
