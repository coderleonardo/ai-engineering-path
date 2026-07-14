# Vector Databases

## Why Retrieval-Augmented Generation (RAG)

An LLM's knowledge is frozen at training time and limited to its context window — it cannot answer
questions about a private document it has never seen. RAG works around this without retraining anything:
relevant chunks of your own documents are retrieved at query time and inserted directly into the prompt as
context, so the model answers from text it can actually see, not from memorized (and possibly outdated or
hallucinated) knowledge. Vector databases are the retrieval half of that pipeline.

## What They Are

Vector Databases are systems designed to store vectors.

In the context of natural language processing and LLMs, we typically store high-dimensional representations of objects in these databases, such as images, texts, or complex information, via embeddings.

This form of storage is efficient because it enables:

- Representing our data in a multidimensional space

- Similarity search capabilities

- Efficient indexing of large data volumes, increasing search speed

```
stored documents (embedded once):
  doc A: "Our return policy allows refunds within 30 days."  -> [0.12, 0.87, ...]
  doc B: "Shipping takes 3-5 business days."                 -> [0.55, 0.10, ...]

query (embedded at search time):
  "How long do I have to return an item?"                    -> [0.15, 0.83, ...]

nearest by cosine similarity -> doc A
```

The query never has to share exact words with a document — it just has to land near it in vector space,
which is what lets retrieval work on paraphrased or loosely-worded questions.