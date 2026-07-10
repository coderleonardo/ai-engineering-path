# LangChain — Core Concepts

LangChain is a framework for building LLM-powered applications by composing smaller, reusable pieces —
prompts, models, memory, retrieval — into a single pipeline, rather than hand-wiring API calls together
for every new application.

Reference: https://python.langchain.com/docs/get_started/introduction

## Prompt Templates

A structured way to format input for an LLM: a template string with placeholder variables that get filled
in dynamically, instead of hardcoding complete prompts. This makes prompt construction consistent and
programmatic — the same template can be reused across many inputs, and the placeholders make explicit
exactly what varies between calls. See also
[module 05](../05-prompt-engineering/prompt_engineering_basics.md) for the general prompt-phrasing
strategies a template's static text should follow.

## Chains

A **chain** is a sequence of operations that processes an input and produces an output by combining LLMs,
other chains, and specialized tools/utilities. Chains are LangChain's basic unit of composition — building
blocks for more advanced constructs like agents.

- **LLMChain** — the simplest chain: a prompt template plus an LLM. It provides a structured interface for
  passing input into the template, sending the result to the model, and returning the generation.
- **SimpleSequentialChain** — runs a fixed sequence of chains in order, passing the single output of each
  step directly as the single input to the next. Straightforward, but limited to strictly linear,
  single-value handoffs.
- **SequentialChain** — a more flexible sequential chain: each step's inputs/outputs are named keys rather
  than an implicit single value, so multiple chains can read and write distinct named variables and later
  steps can draw on any earlier output, not just the immediately preceding one.

## Memory

**Memory** components let chains, agents, and other constructs store and retrieve information from
previous inputs, outputs, and intermediate state — without memory, every call to a chain is stateless and
has no awareness of prior turns.

- **ConversationBufferMemory** — stores the full raw conversation history and replays all of it back into
  the prompt on each turn. Simple and complete, but the prompt grows without bound as a conversation
  continues.
- **ConversationBufferWindowMemory** — keeps only the last *k* interactions, discarding older ones. This
  bounds prompt size and avoids overloading the model with irrelevant history, at the cost of losing
  context beyond the window.
- **ConversationChain** — a chain specialized for multi-turn dialogue: it comes with a default
  conversational prompt template and integrates directly with a memory component, abstracting away the
  bookkeeping of formatting history into each new prompt.

## Retrieval-Augmented Generation

LangChain wires retrieval and generation together through a dedicated chain type (commonly
`RetrievalQA`) that takes a retriever (backed by a vector store) and an LLM, and answers a query by first
retrieving relevant documents and then generating a response grounded in them — the same RAG pattern
covered conceptually in [module 05](../05-prompt-engineering/vector_databases.md), just expressed as a
LangChain chain instead of a hand-assembled prompt.

The **"stuff"** chain type is the simplest way to combine retrieved documents with a question: it
concatenates ("stuffs") every retrieved document directly into the prompt in one shot, appropriate as long
as the retrieved context comfortably fits the model's context window.
