Transformers split into three architecture families, distinguished by which direction of context each one
attends to and what that makes them good at.

## BERT (Bidirectional Encoder Representations from Transformers) — Encoder-only

Bidirectional model that uses information both to the left and right of a word in a sequence to perform a specific task.

Useful for problems such as sentiment classification and named entity extraction.

```
input:  "The movie was [MASK] good."
output: "really"   (predicted using context from both sides of the masked word)
```

More detailed explanation: https://huggingface.co/blog/bert-101

## GPT (Generative Pre-Trained Transformer) — Decoder-only

Focused on text generation, GPT uses the context to the left to predict the next word in a sequence.

Useful for text generation problems.

```
input:  "Once upon a"
output: "time"   (predicted using only the tokens that came before it)
```

Technical report on GPT-4: https://cdn.openai.com/papers/gpt-4.pdf

## T5 (Text-To-Text Transfer Transformer) — Encoder-Decoder

Combines both halves: an encoder builds a bidirectional representation of the full input, and a decoder
generates output tokens conditioned on that representation plus whatever it has generated so far.

Useful for text-to-text tasks where the output is a transformation of the input rather than a pure
continuation — translation, summarization, question answering.

```
input:  "translate English to French: Hello, how are you?"
output: "Bonjour, comment ça va ?"
```

Reference: https://blog.research.google/2020/02/exploring-transfer-learning-with-t5.html