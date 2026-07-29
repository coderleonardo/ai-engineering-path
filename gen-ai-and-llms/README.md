# Gen AI & LLMs — Cookbook

Personal coursework notes, reorganized as a fast
reference: jump to the module that covers the concept you need, skim the one-line summary below, and
follow the links for the full explanation or copy-paste-ready code.

## How This Is Organized

Each numbered folder is one lecture/project from the course (`06-07-08-...` bundles three lectures that
share one project). Inside a folder:

- **`.md` files** are theory notes — concept explanation, a Mermaid diagram for any multi-step pipeline,
  and (where useful) worked examples or pseudocode.
- **`.ipynb`/`.py` files** are runnable reference implementations — modernized to current library APIs,
  commented on *why* each step exists, and safe to copy into another project.
- **`README.md`** (where present) is the fuller pipeline walkthrough a module's reference notebook
  distills — hyperparameters used, what each config choice trades off, etc.
- A `dsa/` subfolder, if present, is the original course-provided material (gitignored — see the root
  `CLAUDE.md`). Everything tracked and linked below is the cleaned-up, dependency-current version of it.

Folder numbers skip around (no `01`/`02`/`11`/`15` here) — those are course lectures without a
corresponding hands-on module in this repo, not missing content.

## Modules

### [03 — Transformer Architectures & Evaluation Metrics](./03-transformers-and-llms-p1/)

Theory-only: the three transformer families (encoder-only/BERT, decoder-only/GPT, encoder-decoder/T5)
and the two families of evaluation metrics (classification vs. generation) that the rest of the course
reuses. `README.md` maps which later module uses which architecture/metric.

**Files:** [`transformers_architectures.md`](./03-transformers-and-llms-p1/transformers_architectures.md),
[`metrics_to_evaluate_llms.md`](./03-transformers-and-llms-p1/metrics_to_evaluate_llms.md)
**Reference:** [BERT 101](https://huggingface.co/blog/bert-101) · Data Science Academy — AI Engineering course

### [04 — Text Generation with GPT-2](./04-transformers-and-llms-p2/)

First hands-on generation: load a pretrained decoder-only GPT-2 and drive `.generate()` with beam search
— no fine-tuning, no training loop, just the encode → generate → decode pattern every later module builds
on.

**Files:** [`text_generation.ipynb`](./04-transformers-and-llms-p2/text_generation.ipynb)
**Reference:** [gpt2-large](https://huggingface.co/gpt2-large)

### [05 — Prompt Engineering & RAG](./05-prompt-engineering/)

How to phrase prompts effectively, plus a first end-to-end RAG pipeline: PDFs → embeddings → Chroma →
retrieval → an LLM answer grounded in the retrieved context, shown both the legacy-chain and LCEL way.

**Files:** [`prompt_engineering_basics.md`](./05-prompt-engineering/prompt_engineering_basics.md),
[`vector_databases.md`](./05-prompt-engineering/vector_databases.md),
[`rag_and_chats.ipynb`](./05-prompt-engineering/rag_and_chats.ipynb)
**Reference:** [LangChain docs](https://python.langchain.com/docs/get_started/introduction) · Data Science
Academy — AI Engineering course

### [06-07-08 — Transfer Learning & Fine-Tuning (QLoRA on Llama-2)](./06-07-08-transfer-learning-fine-tuning/)

PEFT/LoRA/QLoRA theory with a worked parameter-count example, then a full QLoRA fine-tune of Llama-2-7b
that frames sentiment classification as *generation* (the model writes the word "Positive"/"Negative"),
followed by merging the LoRA adapters back into the base model.

**Files:** [`fine_tuning.md`](./06-07-08-transfer-learning-fine-tuning/fine_tuning.md),
[`qlora_sentiment_finetuning.ipynb`](./06-07-08-transfer-learning-fine-tuning/qlora_sentiment_finetuning.ipynb)
**Reference:** [LoRA paper](https://arxiv.org/abs/2106.09685) ·
[QLoRA paper](https://arxiv.org/abs/2305.14314) ·
[Llama-2-7b-chat-hf](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf)

### [09 — Legal Assistant (FLAN-T5 Fine-Tuning for Q&A)](./09-legal-assistant-llm-finetuning/)

Full (non-PEFT) fine-tuning of an encoder-decoder model on legal question/answer pairs, with ROUGE wired
into the training loop for evaluation.

**Files:** [`README.md`](./09-legal-assistant-llm-finetuning/README.md),
[`flan_t5_legal_qa_finetuning.ipynb`](./09-legal-assistant-llm-finetuning/flan_t5_legal_qa_finetuning.ipynb)
**Reference:** [T5 transfer learning blog](https://blog.research.google/2020/02/exploring-transfer-learning-with-t5.html)
· [flan-t5-base](https://huggingface.co/google/flan-t5-base) ·
[ymoslem/Law-StackExchange](https://huggingface.co/datasets/ymoslem/Law-StackExchange)

### [10 — Customer Service Bot (Falcon-7B QLoRA)](./10-customer-service-bot/)

The same QLoRA mechanism as module 06-07-08 applied to a decoder-only chat model for support Q&A, with
BLEU used for evaluation this time instead of ROUGE.

**Files:** [`README.md`](./10-customer-service-bot/README.md),
[`falcon_qlora_customer_support_finetuning.ipynb`](./10-customer-service-bot/falcon_qlora_customer_support_finetuning.ipynb)
**Reference:** [falcon-7b](https://huggingface.co/tiiuae/falcon-7b) ·
[QLoRA / bitsandbytes](https://huggingface.co/blog/4bit-transformers-bitsandbytes) ·
[PEFT docs](https://huggingface.co/docs/peft)

### [12 — LangChain Part 1 (Chains, Memory & RAG)](./12-langchain-p1/)

Prompt templates, chain composition (both legacy `Chain` classes and current LCEL), conversational memory
(legacy memory classes and the current checkpointer pattern), and Chroma as a concrete vector store.

**Files:** [`langchain.md`](./12-langchain-p1/langchain.md), [`chroma.md`](./12-langchain-p1/chroma.md),
[`langchain_chains_memory_rag.ipynb`](./12-langchain-p1/langchain_chains_memory_rag.ipynb),
[`README.md`](./12-langchain-p1/README.md)
**Reference:** [LangChain docs](https://python.langchain.com/docs/get_started/introduction) ·
[Chroma docs](https://docs.trychroma.com)

### [13 — Instruction Fine-Tuning (Llama-2 for Medical Q&A)](./13-langchain-project/)

What instruction fine-tuning is and why it turns a raw text-completer into something that follows
instructions, then a concrete QLoRA instruction-tune of Llama-2 wrapped as a LangChain LLM
(`HuggingFacePipeline`) inside an LCEL chain.

**Files:** [`instruction_finetuning.md`](./13-langchain-project/instruction_finetuning.md),
[`medical_llm_instruction_finetuning.ipynb`](./13-langchain-project/medical_llm_instruction_finetuning.ipynb)
**Reference:** [nlpie/Llama2-MedTuned-Instructions](https://huggingface.co/datasets/nlpie/Llama2-MedTuned-Instructions)
· [Llama-2-7b-chat-hf](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf)

### [14 — Conversational Agents](./14-conversational-agents/)

A tool-using chat agent (`create_agent` + a checkpointer) that decides per-turn whether to call a web
search tool, wired into a Streamlit UI with thread-scoped memory — plus a from-scratch Docker primer
(images vs. containers, layer caching, `Dockerfile` instruction semantics) tied to this module's own
`Dockerfile`.

**Files:** [`streamlit_search_agent.py`](./14-conversational-agents/streamlit_search_agent.py),
[`docker.md`](./14-conversational-agents/docker.md)
**Reference:** [LangChain agents](https://python.langchain.com/docs/get_started/introduction) ·
[Docker get-started docs](https://docs.docker.com/get-started/)

### [16 — Vector Databases](./16-vector-databases/)

How an embedding is actually produced, the ANN indexing algorithms behind vector search (HNSW, LSH,
Random Projection, Product Quantization) with pseudocode for each, the similarity metrics used to rank
results, and a fully local RAG notebook (Chroma + MMR retrieval + local generation, no external API).

**Files:** [`vector_databases.md`](./16-vector-databases/vector_databases.md),
[`local_rag_knowledge_base.ipynb`](./16-vector-databases/local_rag_knowledge_base.ipynb)
**Reference:** [Chroma docs](https://www.trychroma.com) ·
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) ·
[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

### [17 — Deploy](./17-deploy/)

LLM deployment strategy, production monitoring, and data security/privacy in depth, plus what
"reasoning" actually means for an LLM — illustrated concretely by training a tiny GPT-2-architecture
model from scratch on 1-digit addition (no pretraining, no chain-of-thought) and deploying it through a
minimal Gradio app.

**Files:** [`deployment.md`](./17-deploy/deployment.md), [`reasoning.md`](./17-deploy/reasoning.md),
[`tiny_addition_llm_training.ipynb`](./17-deploy/tiny_addition_llm_training.ipynb),
[`tiny_addition_llm_gradio_deploy.ipynb`](./17-deploy/tiny_addition_llm_gradio_deploy.ipynb)
**Reference:** [Towards Reasoning in Large Language Models: A Survey](https://arxiv.org/pdf/2212.10403) ·
[Gradio docs](https://www.gradio.app)

### [18 — RAG](./18-rag/)

A complete, generalizable RAG application: multi-format document ingestion (PDF/TXT/DOCX/PPTX), chunking
strategy, a Qdrant-backed retrieval API with citation/grounding, and a Streamlit client — plus
project-agnostic reference notes on both RAG application engineering and the broader LLMOps lifecycle
(evaluation, prompt versioning, RAG-specific data lifecycle, cost/security governance).

**Files:** [`rag.md`](./18-rag/rag.md), [`llmops.md`](./18-rag/llmops.md),
[`document_indexer.py`](./18-rag/document_indexer.py), [`rag_api.py`](./18-rag/rag_api.py),
[`rag_streamlit_app.py`](./18-rag/rag_streamlit_app.py), [`README.md`](./18-rag/README.md) (run instructions)
**Reference:** [Qdrant docs](https://qdrant.tech/documentation/) ·
[OpenAI API docs](https://platform.openai.com/docs)
