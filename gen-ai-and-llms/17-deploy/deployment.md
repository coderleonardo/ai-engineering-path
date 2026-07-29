# LLM Deployment — Concepts & Notes

Training or fine-tuning a model (as in
[`06-07-08-transfer-learning-fine-tuning`](../06-07-08-transfer-learning-fine-tuning/fine_tuning.md),
[`09-legal-assistant-llm-finetuning`](../09-legal-assistant-llm-finetuning/README.md)) produces a
checkpoint on disk — it isn't usable by anyone until it's deployed: made reachable, kept running, and
kept correct over time. This module's own
[`tiny_addition_llm_gradio_deploy.ipynb`](./tiny_addition_llm_gradio_deploy.ipynb) is the smallest
possible instance of that: `save_pretrained` -> `from_pretrained` -> `gr.Interface(...).launch()`, with no
load balancing, no auth, no monitoring, running on a single machine. It's referenced throughout below as
the "toy-scale" anchor for what each production concept would add on top of it.

## Deployment Strategies

### 1. Choosing the Right Infrastructure: Cloud or On-Premises?

- **Cloud** (AWS, GCP, Azure, or managed inference platforms) — no upfront hardware cost, elastic
  capacity (scale GPU instances up/down with demand), and managed services for the surrounding pieces
  (load balancers, logging, autoscaling) instead of building them by hand. Trade-off: ongoing per-hour
  cost, less control over exact hardware, and data leaving your own network unless carefully configured.
- **On-premises** — full control over hardware and data locality (important under strict data-residency
  or compliance requirements), and a fixed cost once hardware is bought rather than a variable one. Trade-
  off: capacity is fixed by whatever hardware is on hand — a traffic spike beyond that requires buying
  more GPUs, not just paying more, and someone has to physically maintain the machines.
- In practice, many teams land on a **hybrid** split: sensitive data/inference kept on-premises, burst
  capacity or non-sensitive workloads handled in the cloud.

### 2. Model Optimization

Making a trained model cheaper/faster to run in production, ideally without materially hurting accuracy:

- **Quantization** — storing/computing weights at lower numeric precision (e.g. FP32 -> INT8 or 4-bit)
  instead of full precision. Shrinks memory footprint and speeds up inference, at the cost of some
  precision loss. QLoRA (see `fine_tuning.md`) already applies this idea at *training* time; the same
  principle applies again at *inference* time for a model that was trained in full precision.
- **Pruning** — removing weights/neurons/attention heads that contribute little to the model's output,
  shrinking the model directly rather than just its numeric representation.
- **Distillation** — training a smaller "student" model to mimic a larger "teacher" model's outputs. The
  student ends up cheaper to run than the teacher while recovering much of its behavior — different from
  quantization/pruning, which shrink the *same* model rather than train a new, smaller one.

### 3. Load Management

- **Load balancing** — distributing incoming inference requests across multiple model server instances
  instead of overwhelming one, the same general idea as load balancing any web service, applied to
  (typically GPU-bound, latency-sensitive) inference requests specifically.
- **Auto-scaling** — automatically adding/removing model server instances based on current traffic, so
  capacity roughly tracks demand instead of being sized for worst case (wasteful) or average case (falls
  over under spikes).

### 4. Security and Privacy

- **Data anonymization** — stripping or masking personally identifiable information (PII) from data
  before it reaches the model or gets logged, so a prompt/response pipeline doesn't become an
  unintentional PII store.
- **Audit** — logging who accessed the model, with what inputs, and when, so usage is traceable after the
  fact — necessary both for debugging and for compliance investigations.

(See also the dedicated [Data Security and Data Privacy](#data-security-and-data-privacy) section below —
that section covers the full data lifecycle; this bullet is specifically about protecting data *in the
serving path*.)

### 5. Integrations

- **API integration** — exposing the model behind a stable API contract (REST/gRPC) so client
  applications can call it without knowing anything about how it's hosted. `gr.Interface` in this
  module's deploy notebook is a minimal example: it wraps a Python function in both a web UI *and* an
  implicit HTTP API, without either side needing to know about the other.
- **Monitoring** — wiring the deployment into whatever observability stack the rest of the org already
  uses (metrics, logs, traces), rather than treating the model server as a black box — see
  [Monitoring Models in Production](#monitoring-models-in-production) below for what to actually track.

### 6. Ethics

Deployment is where model behavior actually reaches users, so this is where ethical risks (biased
outputs, harmful content, misuse) become concrete rather than theoretical. Concretely: define what the
model must refuse or flag before launch, not after an incident; keep a human-review path for edge cases;
and revisit these policies as real usage patterns emerge, since misuse patterns are rarely fully
predictable in advance.

### 7. Caching and State Management

- **Caching** — storing results for repeated/similar inputs so identical requests don't re-run inference,
  cutting both cost and latency. Especially effective when a meaningful fraction of traffic repeats
  (common prompts, FAQ-style queries).
- **State management** — for anything beyond single-turn inference (e.g. a multi-turn conversational
  agent, like [module 12](../12-langchain-p1/langchain.md#memory)'s checkpointer-based memory), deployment
  also has to decide *where* that state lives — in-memory per instance (breaks if a request is routed to
  a different instance next turn), or in a shared store all instances can reach.

### 8. Continuous Improvement

- **Updates** — rolling out a new model version (retrained, fine-tuned, or just reconfigured) without
  breaking clients depending on the current one — typically via versioned endpoints or a gradual
  (canary/blue-green) rollout rather than a hard cutover.
- **Feedback loop** — capturing signals from real usage (explicit ratings, implicit signals like
  reformulated queries or abandoned sessions) and feeding them back into future training/fine-tuning
  data, closing the loop between "deployed" and "improved."

## Monitoring Models in Production

Deployment isn't a one-time event — a model serving live traffic needs ongoing observation, because
unlike traditional software, its "correctness" can silently degrade even when the code never changes.

1. **Model performance monitoring** — tracking accuracy/latency/throughput metrics over time, so a
   regression is caught from the metrics rather than from user complaints.

2. **Data validation: data quality and data drift** — checking that incoming production inputs still
   resemble what the model was trained/evaluated on. **Drift** specifically means the input distribution
   has shifted over time (e.g. a customer-support bot suddenly getting a wave of questions about a topic
   it never saw in training) — the model's weights haven't changed, but its effective accuracy on
   *today's* traffic can drop anyway, since it's now being asked things unlike what it learned from.

3. **Model retraining** — once drift or performance decay is detected, refreshing the model on newer data
   is the usual fix; this loops back into "Updates" above.

4. **Dependencies management**
   - **Model version** — pinning exactly which checkpoint is serving traffic, so a bug report can be
     traced back to a specific version instead of "whatever's currently deployed."
   - **Model environment** — pinning the runtime (framework version, CUDA version, Python version) the
     model was validated against; the same weights can behave differently under a different
     `transformers`/`torch` version.

5. **Data privacy and security** — the serving-time counterpart of item 4 in Deployment Strategies above:
   monitoring who accessed what, and confirming no unintended data (PII, proprietary content) is leaking
   into logs or being persisted longer than necessary.

6. **Feedback loop** — same idea as in Deployment Strategies, but framed as an ongoing monitoring signal:
   feedback volume/sentiment itself is something worth tracking as a metric, not just a one-off input to
   retraining.

7. **Compliance** — verifying the deployed system still satisfies whatever regulatory requirements apply
   (e.g. GDPR, HIPAA, industry-specific rules) on an ongoing basis, since compliance obligations don't end
   at launch.

## Data Security and Data Privacy

This is the dedicated, full-lifecycle treatment of data handling — the bullets under "Security and
Privacy" and "Data privacy and security" above are specifically about the serving path; this section is
about data at every stage a deployed system touches it:

- **Encryption** — in transit (TLS between clients, load balancers, and model servers) and at rest (any
  logs, caches, or fine-tuning data stored on disk), so a network intercept or a stolen disk doesn't
  directly expose raw data.
- **Access control** — restricting who/what can query the model, view logs, or access training data, via
  authentication and least-privilege permissions, rather than a single shared credential everyone uses.
- **Data minimization and retention** — logging only what's actually needed for debugging/monitoring, and
  deleting it after a defined retention window, rather than keeping every prompt/response indefinitely
  "just in case."
- **Anonymization/pseudonymization** — stripping or replacing PII in any data that does need to be
  retained (e.g. for retraining), so a leak of that data doesn't directly expose individuals.
- **Regulatory alignment** — data-handling practices need to match whatever regime applies to the data in
  question (GDPR in the EU, HIPAA for health data, etc.) — this is the data-handling half of the
  "Compliance" item under monitoring above.

## Challenges

1. **Cost and infrastructure** — GPU inference is expensive at scale, and the cloud-vs-on-premises
   trade-off above rarely has a clean answer; costs also shift over the model's lifetime (training cost is
   one-time, serving cost is ongoing and scales with traffic).

2. **Data management** — sourcing, cleaning, and governing the data used both for training/fine-tuning and
   for the ongoing monitoring described above is a continuous effort, not a one-time setup step.

3. **Model performance and maintenance** — keeping a model accurate over time requires the drift-detection
   and retraining loop above; skipping it means performance decay goes unnoticed until it's visible to
   users.

4. **Integration complexity in existing infrastructure** — a model server rarely stands alone; it has to
   fit into existing auth, logging, API gateway, and deployment pipelines, which is often more engineering
   effort than building the model server itself.

5. **Ethics** — the same concerns as item 6 in Deployment Strategies, but as an ongoing challenge rather
   than a one-time checklist: new misuse patterns and edge cases keep surfacing after launch, not just
   before it.
