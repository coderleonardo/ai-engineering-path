# Q&A — Course Questions and Answers

Quick questions and answers on course topics, for review. Every answer is grounded in content already
present in this repository — links point to the original note with the full explanation.

## Transformer Architectures

### What's the big difference between how BERT and GPT-3 are trained?

BERT (encoder-only) is trained with **Masked Language Modeling**: it masks ~15% of tokens and predicts
the original token using **bidirectional** context (both sides), plus Next Sentence Prediction. GPT-3
(decoder-only) is trained with **Causal/Autoregressive Language Modeling**: it predicts the next token
using only **left** context — the exact same task used in training is literally repeated token by token
at generation time.

The difference that matters most in practice: BERT was designed for **fine-tuning** per task
(classification, NER, with an extra head); GPT-3, through scale (175B parameters), made **in-context
learning** viable — solving new tasks from just examples in the prompt, without updating any weights.

See [transformers_architectures.md](./03-transformers-and-llms-p1/transformers_architectures.md).

### Can I say BERT is better suited for classification/extraction while GPT is better suited for translation/composition?

Only partly. BERT (encoder-only) for classification/extraction is correct — it produces contextual token
representations rather than generating text. But **translation** is the canonical example given for
**T5 (encoder-decoder)** in the course notes, not GPT: T5 uses a bidirectional encoder to "read" the
entire source sentence before generating the translation, something a pure decoder-only model doesn't
have. GPT-3/4 can translate well via prompting, but that's an emergent capability of scale, not the
architecture's natural fit for the task.

| Model | Strong at |
|---|---|
| BERT (encoder-only) | Classification, extraction, NER, sentiment |
| T5 (encoder-decoder) | Translation, summarization, QA — text-to-text transformation |
| GPT (decoder-only) | Free-form composition, continuation, open-ended generation, chat |

See [transformers_architectures.md](./03-transformers-and-llms-p1/transformers_architectures.md).

### In a Transformer, is the output layer linear or non-linear?

The final projection ("LM Head") is **linear**: a plain matrix multiplication that projects the hidden
state into vocabulary-sized logits, with no activation function. But that layer is followed by
**softmax**, which is **non-linear**, converting the raw logits into a probability distribution over the
vocabulary.

| Step | Nature |
|---|---|
| Hidden state → logits projection (LM Head) | Linear |
| Logits → probabilities (Softmax) | Non-linear |

See [module 04 README](./04-transformers-and-llms-p2/README.md#1-tokenizer-and-model).

### Did DeepMind use Transformers to defeat game players?

Not as a general rule — it depends on the system. **AlphaGo** and **AlphaZero** used CNN + Monte Carlo
Tree Search, and predate the Transformer paper (2017), so they couldn't have used that architecture at
all. The correct case is **AlphaStar** (StarCraft II, 2019): its network includes a self-attention
(Transformer) encoder to process the game's units as a set of entities, combined with an LSTM. More
recently, **Gato** (2022) is a generalist agent built entirely on a Transformer, able to play hundreds of
Atari games.

## Tokenization

### Does the `tokenizers` package handle text tokenization?

Yes. `tokenizers` (Hugging Face, written in Rust) implements the tokenization algorithms themselves (BPE,
WordPiece, Unigram) with a focus on performance. In practice you rarely call it directly — the
`transformers` package's `AutoTokenizer` uses it under the hood, also handling vocabulary, special tokens
(`[CLS]`, `[SEP]`, `<pad>`), and padding/truncation.

### Which package is designed for fast, efficient tokenization: `tokenizers` or `sentencepiece`?

**`tokenizers`** — implemented in Rust with native parallelization, explicitly built for speed.
`sentencepiece` (Google) solves a different problem: it's a **language-independent** tokenizer that
operates directly on raw text (no whitespace-based pre-tokenization needed), useful for languages like
Japanese/Chinese, and is lossless (it can reconstruct the original text exactly). It's used by T5, Llama,
ALBERT; its edge is cross-language generalization, not raw speed.

## Fine-Tuning and PEFT

### What's the first step in the fine-tuning process?

**Pre-trained model selection** — choosing an architecture aligned with the task's domain. This comes
before data preparation because it determines the tokenizer, the expected input format, and whether an
encoder-only, decoder-only, or encoder-decoder model is needed.

See [fine_tuning.md](./06-07-08-transfer-learning-fine-tuning/fine_tuning.md#fine-tuning-with-your-own-data).

### Do LoRA and QLoRA eliminate the need for adapters?

No — it's the opposite: **LoRA and QLoRA are themselves a type of adapter**. "Adapter" is the generic
term for freezing pre-trained weights and training a small extra module. LoRA implements this by injecting
two trainable low-rank matrices ($A$, $B$): $h = W_0 x + (BA)x$. QLoRA just adds 4-bit quantization (NF4)
to the frozen base model, without changing that logic. What LoRA/QLoRA eliminate is the need to update
**all** of the model's weights or keep a full copy per task — not the concept of an adapter itself.

See [fine_tuning.md](./06-07-08-transfer-learning-fine-tuning/fine_tuning.md#lora-low-rank-adaptation).

### What's the name of the fine-tuning technique where we only adjust some layers of the pre-trained model?

**Selective Layer Freezing**: freeze most of the model and train only a subset of layers (typically the
last ones). A related variant is **Gradual Unfreezing**: start by training only the last layer and
progressively unfreeze deeper layers as training goes on.

See [fine_tuning.md](./06-07-08-transfer-learning-fine-tuning/fine_tuning.md#peft-parameter-efficient-fine-tuning).

### Does Selective Layer Freezing classify as differential learning rate, parameter adjustment, layer addition, or regularization?

**Parameter adjustment** (Parameter Subset Adjustment) — this maps directly to that strategy in the
course notes: instead of updating all parameters θ, you restrict the update to a subset Δθ (the unfrozen
layers), keeping the rest frozen. It's not "differential learning rate" (which gives layers different
LRs while all remain trainable), nor "layer addition" (inserts new modules while keeping the original
model fully frozen), nor "regularization" (dropout, weight decay — fights overfitting, doesn't decide
which layers train).

### What's a crucial factor for making a dataset effective for supervised fine-tuning?

**Label quality and consistency.** In supervised fine-tuning the model learns exactly the pattern present
in the input→output pairs; noisy or inconsistent labels get learned as if they were signal, and that
noise doesn't dilute away with more data the way one might expect — especially in fine-tuning, with few
epochs and little data compared to pretraining. Complementary factors: representativeness of the real
usage distribution, class balance, consistent formatting, and sufficient size (though quality typically
outweighs quantity, particularly under PEFT/LoRA regimes).

### What is SFTTrainer and what's its main characteristic?

It comes from the **TRL** library (Hugging Face). Its core characteristic: it abstracts the supervised
training loop for instruction-response formatted data, treating the task as pure **causal language
modeling** (next-token prediction) over already-formatted text — no separate classification head needed.
It expects a pre-formatted text column (`dataset_text_field`) or a `formatting_func`, and integrates
natively with PEFT (LoraConfig) and quantization (bitsandbytes/QLoRA).

See [module 06-07-08 README](./06-07-08-transfer-learning-fine-tuning/README.md#3-training-configuration--two-configs-one-used).

### Which package do we use to optimize the process around multi-GPU, TPU, and FP16?

**`accelerate`** (Hugging Face). It abstracts distributing training across multiple GPUs, TPUs, and
manages mixed precision (FP16/BF16) without requiring the training loop to be rewritten. The
`Trainer`/`SFTTrainer` from `transformers`/`trl` already use it under the hood — typically you just pass
`fp16=True`/`bf16=True` in the training arguments.

### What is instruction fine-tuning and why is it needed?

It's a form of Supervised Fine-Tuning where the training data is structured as
**(instruction, input, response)** triples. A base model, pretrained only with next-token prediction on
raw text, is good at *continuing* text but has no notion of "answer what was asked" — instruction
fine-tuning is what turns that raw text-completer into something that behaves like an assistant,
following instructions, respecting output format, and serving as the foundation for further alignment
(RLHF/DPO).

See [instruction_finetuning.md](./13-langchain-project/instruction_finetuning.md).

## Domain-Specific Models

### Which model should we use to classify sentiment in scientific articles: DataBERT, SciBERT, SuperBet, or something else?

**SciBERT** — a BERT variant pretrained from scratch on ~1.14M scientific papers (Semantic Scholar), with
its own vocabulary (SciVocab) adapted to technical language. This prevents scientific terms from being
split into poorly informative subwords, as would happen with a generic BERT trained on Wikipedia/books.
"DataBERT" and "SuperBet" don't correspond to any known real models.

### Does SciBERT use the same masking step as BERT to predict the masked word?

Yes — the pretraining objective is identical: **Masked Language Modeling** (masks tokens and predicts
them using bidirectional context) plus Next Sentence Prediction, following the same procedure as the
original BERT. What differentiates SciBERT isn't *how* it learns, but *what it learns from*: a scientific
corpus and its own vocabulary, instead of Wikipedia/BookCorpus.

### Were GPT-Neo and GPT-NeoX created to be better than GPT-3?

Not in terms of quality/scale — they were created by EleutherAI to solve an **access** problem: when
GPT-3 launched, OpenAI didn't open the weights, only a paid API. GPT-Neo (1.3B/2.7B) and GPT-NeoX (20B)
are orders of magnitude smaller than GPT-3 (175B) and don't outperform it on benchmarks — the goal was to
democratize access to GPT-like models with open weights, trained on the open **The Pile** dataset, not to
beat GPT-3 on raw performance.

## Model Evaluation

### What's the difference between BLEU and ROUGE?

Both compare generated text against a reference via n-gram overlap, but with different emphasis:
**BLEU** leans **precision** ("how much of what was generated is relevant"), with a brevity penalty
against outputs that are too short — it originates from machine translation. **ROUGE** leans **recall**
("how much of the reference was captured"), reporting precision/recall/F1 for each variant (ROUGE-N by
n-grams, ROUGE-L by Longest Common Subsequence) — it originates from summarization, but is also used for
QA/answer generation.

See [metrics_to_evaluate_llms.md](./03-transformers-and-llms-p1/metrics_to_evaluate_llms.md).

### What does Perplexity measure?

It evaluates how well the probability predicted by the model aligns with the actual distribution of the
data — the **lower**, the better. It's the typical metric for evaluating a language model on its own
(not a downstream task), measuring how "surprised" the model is by real text.

See [metrics_to_evaluate_llms.md](./03-transformers-and-llms-p1/metrics_to_evaluate_llms.md#perplexity).

## Reasoning and Prompting

### What is Chain-of-Thought prompting and why does it help?

**Chain-of-thought (CoT) prompting** asks the model to write out its intermediate reasoning steps before
giving a final answer, instead of asking for the answer directly. It works because the model's own
previously-generated tokens become part of the input it conditions on for predicting the next token —
writing out intermediate results exposes the model to information a one-shot "the answer is X" prompt
never gives it. CoT is a *prompting* technique: it doesn't change the model's weights, only what it's
asked to generate before the answer, and it works best on large models with broad pretraining exposure to
worked, step-by-step examples.

See [reasoning.md](./17-deploy/reasoning.md#chain-of-thought-prompting).

### Does fluent text generation imply an LLM is actually reasoning?

No. An LLM is trained to do one thing: predict the next token given everything before it. Most of what
looks like "understanding" is pattern completion learned from massive repetition. **Reasoning** is the
harder case — problems that need several dependent inference steps chained together (arithmetic,
multi-hop QA, logical deduction) — and a model can be excellent at fluent, plausible next-token
prediction while still failing at reasoning, because fluency doesn't imply the intermediate steps are
being computed correctly.

See [reasoning.md](./17-deploy/reasoning.md#what-reasoning-means-here).

## Vector Databases and RAG

### What problem does Retrieval-Augmented Generation (RAG) solve?

RAG grounds an LLM's output in retrieved external data instead of relying solely on what the model
memorized during training — addressing frozen training-time knowledge, the model's limited context
window, and hallucination risk. A vector database implements the retrieval half of the pipeline:
documents are embedded and stored once, offline; at query time, the question is embedded and the nearest
stored chunks are pulled back and inserted into the prompt before generation.

See [vector_databases.md](./16-vector-databases/vector_databases.md#vector-databases-in-agentic-systems-rag-framework).

### Why do many vector databases default to dot product instead of cosine similarity?

When embeddings are pre-normalized to unit length (a common practice), cosine similarity and dot product
produce the **same ranking** — but dot product is cheaper to compute, since it skips the normalization
step cosine similarity requires. So defaulting to dot product is mathematically equivalent in that case,
and faster at query time over millions of vectors.

See [vector_databases.md](./16-vector-databases/vector_databases.md#retrieving-information).

### What's the difference between HNSW and LSH as vector indexing strategies?

**HNSW (Hierarchical Navigable Small World)** builds a multi-layer graph of vectors: a sparse top layer
with a few long-range links for cheap large jumps, getting progressively denser at lower layers for
precise refinement near the query — a search walks greedily from the top layer down. **LSH
(Locality-Sensitive Hashing)** instead uses hash functions (e.g. random hyperplanes) designed so similar
vectors collide into the same bucket with high probability — a query only has to compare against that
bucket, not the whole collection. Both are indexing strategies (how vectors are *organized* for search),
as opposed to compression strategies like Random Projection or Product Quantization (how vectors are
*represented* more cheaply) — the two categories are often combined in practice.

See [vector_databases.md](./16-vector-databases/vector_databases.md#how-vector-databases-work).
