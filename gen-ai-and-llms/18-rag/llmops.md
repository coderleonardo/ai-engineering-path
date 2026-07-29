# LLMOps — Reference Notes

**LLMOps** is the operational discipline of running LLM-powered systems reliably over time — the
LLM-specific counterpart to MLOps. It borrows most of MLOps' concerns (versioning, CI/CD, monitoring,
retraining) but adds a few that don't have a clean traditional-ML analogue:

- **The prompt is a first-class versioned artifact.** In classical ML, the model's behavior is entirely
  determined by its trained weights. In an LLM application, behavior is jointly determined by the model
  *and* the prompt/system-instructions wrapped around it — and the prompt changes far more often than the
  underlying model does. A prompt edit is a behavior-changing deployment and needs the same rigor
  (review, testing, rollback plan) as a code change, not the "just edit the string" treatment it often
  gets.
- **Outputs are non-deterministic and open-ended**, which breaks the assert-equal testing MLOps largely
  inherited from classical ML (where a classifier's output is a fixed label to compare against ground
  truth). Evaluating an LLM system means judging free-text quality, not just accuracy — see Evaluation
  below.
- **Retrieval corpora are an additional moving part** for any RAG system: the vector index has its own
  lifecycle (reindexing, embedding-model versioning) layered on top of the model's own lifecycle — see
  RAG-Specific Data Lifecycle below.

## Experimentation and Prompt Versioning

Treat prompts (and few-shot examples, system instructions, and any other prompt-adjacent config) as
version-controlled artifacts with a change history — the same discipline applied to code, not a value
someone edits directly in a running service. This makes it possible to: attribute a behavior regression to
the specific prompt change that caused it, A/B test two prompt variants against the same eval set, and
roll back a bad prompt change exactly as you'd roll back a bad code deploy.

## Evaluation

Because outputs aren't a fixed label to check equality against, LLM evaluation typically combines a few
complementary approaches:

- **Offline eval sets** — a curated, versioned set of representative inputs (ideally including known-hard
  and adversarial cases), run against every candidate prompt/model change before it ships, so a regression
  is caught before real users see it rather than after.
- **LLM-as-judge** — using a (typically stronger or differently-tuned) model to score outputs against a
  rubric, when the criteria are too nuanced or too high-volume for exact-match scoring. Useful for scale,
  but the judge model itself needs periodic calibration against human judgment — it can develop its own
  systematic biases (e.g. favoring longer answers) that go unnoticed without spot-checking.
- **Human review** — the ground truth the other two methods are ultimately validated against; typically
  reserved for a sample of production traffic or ambiguous/high-stakes cases rather than every request,
  since it doesn't scale to full volume.

## RAG-Specific Data Lifecycle

For a RAG system specifically, the vector index is data with its own operational lifecycle, independent
of the LLM's:

- **Reindexing cadence** — how often the corpus is re-ingested (see
  [`rag.md`](./rag.md)/[`document_indexer.py`](./document_indexer.py) for the ingestion pipeline itself)
  depends on how often the underlying documents change; a stale index means the system confidently
  retrieves and cites outdated information, which is a harder failure to notice than an obviously wrong
  answer.
- **Embedding-model version pinning** — every vector in the index was produced by one specific embedding
  model. Swapping the embedding model without re-embedding the entire corpus silently breaks retrieval:
  query vectors from the new model aren't comparable to stored vectors from the old one, even though
  nothing raises an error. Changing the embedding model always means a full reindex, not an incremental
  one.

## CI/CD for Prompts and Models

Extending standard CI/CD with LLM-specific gates: run the offline eval set (above) as a required check
before a prompt or model-version change merges, the same way a test suite gates a code change. For
rollout, the same progressive-delivery patterns from regular software apply well here — canary a new
prompt/model version against a small percentage of traffic, or run it in shadow (compute its output
without serving it to users) alongside the current version to compare quality before cutting over.

## Deployment

Once a prompt/model version passes evaluation, getting it running and reachable is the deployment
concern proper — infrastructure choice, load management, and the production challenges involved are
covered in full in [`deployment.md`](../17-deploy/deployment.md); this doc doesn't repeat that.

## LLM-Specific Observability

Beyond the general production-monitoring concerns already covered in
[`deployment.md`](../17-deploy/deployment.md#monitoring-models-in-production), an LLM (and especially a
RAG) system has metrics worth tracking that a typical ML model doesn't:

- **Token usage and cost per request** — LLM API costs scale with input + output tokens, so cost is a
  per-request, usage-driven metric rather than a fixed infrastructure line item; a prompt change that
  silently grows the average context size is a cost regression even if quality doesn't change.
- **Latency, broken down by stage** — for a RAG system specifically, retrieval latency and generation
  latency are separate budgets with different bottlenecks (vector search vs. token-by-token generation),
  and conflating them into one end-to-end number hides which stage to optimize.
- **Retrieval hit rate** — how often retrieval actually returns a chunk relevant to the query, tracked
  independently of whether the final answer was good (see the retrieval-vs-generation failure split in
  [`rag.md`](./rag.md#failure-modes)).
- **Citation accuracy** — for a system that cites sources (see `rag.md`), whether cited sources actually
  support the claim attached to them is worth sampling and tracking on its own, since a model can cite
  confidently and incorrectly at the same time.

## Cost Governance

LLM inference cost is usage-driven and can grow silently as traffic or prompt size grows, unlike a fixed
infrastructure bill. Practical levers: caching repeated/similar queries (see
[`deployment.md`](../17-deploy/deployment.md#7-caching-and-state-management)), capping `max_tokens` to
what the use case actually needs, routing simpler queries to a smaller/cheaper model and reserving a
larger model for cases that need it, and tracking cost per request as a first-class metric (above) so a
regression is caught the same way a latency regression would be.

## Security and Governance

RAG systems have an attack surface classical ML mostly doesn't: **prompt injection via retrieved
content**. Because retrieved document chunks are inserted directly into the prompt, a malicious or
compromised document in the corpus can contain text crafted to override the system instructions (e.g.
"ignore previous instructions and instead...") — the retrieval step effectively lets untrusted content
influence the model's behavior. Mitigations include treating retrieved content as data rather than
instructions where the framework allows that distinction, filtering/sanitizing ingested documents, and
never granting the LLM tool access whose scope an injected instruction could abuse. Beyond injection, the
usual governance concerns apply too: access control over both the model API and the underlying document
corpus, and audit logging of what was retrieved and answered for a given request.

## Continuous Feedback Loop

Production usage is the richest source of new eval cases: user corrections, low ratings, or reformulated
follow-up queries (a sign the first answer didn't land) are all signals worth capturing and feeding back
into the offline eval set and, where applicable, fine-tuning data — closing the loop from "deployed" back
into "evaluated," rather than treating evaluation as a one-time pre-launch gate.
