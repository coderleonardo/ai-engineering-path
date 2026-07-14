Evaluation metrics split into two groups: classification metrics, reused from traditional ML whenever a
model's output reduces to a discrete label, and generation metrics, needed whenever there's no single
"correct" output to match exactly.

## Perplexity

Evaluates how well the probability predicted by the model aligns with the actual distribution of the data.
The lower, the better.

## Accuracy

Proportion of correct predictions made by the model in relation to the total. Used in classification tasks.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

## Precision and Recall

Precision is the proportion of all positive classifications made by the model that are actually positive.

$$\text{Precision} = \frac{TP}{TP + FP}$$

Recall or true positive rate is the proportion of all actual positives that were correctly classified as positive.

$$\text{Recall} = \frac{TP}{TP + FN}$$

## F1-Score

Harmonic combination between precision and recall.

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Worked example

Out of 100 emails, a spam classifier flags 30 as spam. Of those, 25 really are spam (TP) and 5 aren't
(FP). Of the 70 it left alone, 65 really aren't spam (TN) and 5 actually were (FN, missed):

```
accuracy  = (TP + TN) / 100     = (25 + 65) / 100 = 0.90
precision = TP / (TP + FP)      = 25 / (25 + 5)    = 0.83
recall    = TP / (TP + FN)      = 25 / (25 + 5)    = 0.83
F1        = 2 * (P * R)/(P + R) = 2*(0.83*0.83)/(0.83+0.83) = 0.83
```

High accuracy alone would have hidden the fact that the classifier still misses 1 in 6 real spam emails —
precision and recall are what surface that.

## ROC-AUC and PR-AUC

Commonly used in classification tasks, with PR-AUC indicated for imbalanced data.

## BLEU (BiLingual Evaluation Understudy)

Compares generated text against a reference by measuring n-gram precision — for each n-gram size (1
through 4, typically), what fraction of the generated text's n-grams also appear in the reference — then
combines those precisions with a geometric mean. A **brevity penalty** counteracts the fact that shorter
outputs can trivially inflate precision by only emitting "safe," highly overlapping n-grams; the penalty
pulls the score down when the generated text is noticeably shorter than the reference.

BLEU's origin is machine translation, so it suits tasks with a strong emphasis on precise, close-to-
reference phrasing. Applied in [module 10](../10-customer-service-bot/README.md#6-evaluation--bleu).

### Worked example

Same pair used in the ROUGE example below, for a direct comparison:

```
reference: "the cat sat on the mat"
candidate: "the cat was sat on the mat"
```

Clipped n-gram precision at each size (candidate matches capped at how many times an n-gram actually
appears in the reference):

```
BLEU-1 (unigrams):  6/7 = 0.857
BLEU-2 (bigrams):   4/6 = 0.667
BLEU-3 (trigrams):  2/5 = 0.400
BLEU-4 (4-grams):   1/4 = 0.250

geometric mean of the four = 0.489
brevity penalty = 1.0   (candidate is longer than the reference, so no penalty applies)

BLEU = 1.0 * 0.489 = 0.489
```

**Interpretation:** precision stays reasonably high for single words but collapses as n grows, because the
inserted "was" breaks every 3- and 4-gram that crosses it — one misplaced word costs BLEU far more than it
costs ROUGE-1, since BLEU multiplies precisions across all four n-gram sizes instead of scoring unigram
overlap alone.

## ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Measures how much overlap a generated text has with one or more reference texts. It was designed for
summarization but is the standard metric for any generative task with a "ground-truth" text to compare
against — including QA and answer generation. Where BLEU leans precision (how much of what was generated
is relevant), ROUGE leans recall (how much of the reference was captured) — reported alongside precision
and F1 for each variant:

- **Recall** = overlapping units / units in reference — "how much of the reference did we capture?"
- **Precision** = overlapping units / units in candidate — "how much of what we generated is actually relevant?"
- **F1** = harmonic mean of precision and recall

The variants differ only in what counts as a "unit":

| Variant | Unit compared | Captures |
|---|---|---|
| **ROUGE-N** | n-grams (N=1 unigrams, N=2 bigrams, ...) | word/phrase overlap |
| **ROUGE-L** | Longest Common Subsequence (LCS) | in-order overlap, tolerant to gaps/insertions |
| **ROUGE-W** | weighted LCS | like ROUGE-L, but rewards *consecutive* matches over scattered ones |
| **ROUGE-S** | skip-bigrams (any ordered pair, gaps allowed) | looser word-order sensitivity than ROUGE-N |

In practice, HuggingFace's `evaluate` library reports **ROUGE-1, ROUGE-2, ROUGE-L (and ROUGE-Lsum)** by
default; ROUGE-W and ROUGE-S are rarely used outside the original ROUGE paper/toolkit.

### Worked example

```
reference: "the cat sat on the mat"
candidate: "the cat was sat on the mat"
```

**ROUGE-1** (unigram overlap): reference has 6 tokens, candidate has 7. Matched tokens: `the`(×2), `cat`, `sat`, `on`, `mat` → 6 overlapping unigrams.

```
recall    = 6 / 6 = 1.000
precision = 6 / 7 = 0.857
F1        = 2 * (0.857 * 1.000) / (0.857 + 1.000) = 0.923
```

**ROUGE-2** (bigram overlap): reference bigrams = `{the-cat, cat-sat, sat-on, on-the, the-mat}` (5); candidate bigrams = `{the-cat, cat-was, was-sat, sat-on, on-the, the-mat}` (6). Matched: `the-cat, sat-on, on-the, the-mat` → 4.

```
recall    = 4 / 5 = 0.800
precision = 4 / 6 = 0.667
F1        = 2 * (0.667 * 0.800) / (0.667 + 0.800) = 0.727
```

**ROUGE-L** (LCS): the longest common subsequence between the two sentences is `the cat sat on the mat` (length 6) — the extra word `was` in the candidate simply doesn't participate.

```
recall    = 6 / 6 = 1.000
precision = 6 / 7 = 0.857
F1        = 0.923   (same as ROUGE-1 in this example)
```

**Interpretation:** ROUGE-1 shows almost all reference content was reproduced; ROUGE-2 drops because
inserting "was" breaks two bigrams; ROUGE-L confirms the sentence still preserves the reference's word
order despite the insertion. Comparing several variants side-by-side is how you distinguish "captured the
right words" (ROUGE-1) from "captured the right phrasing/order" (ROUGE-L).

Applied in [module 09](../09-legal-assistant-llm-finetuning/README.md#3-evaluation-metric--rouge).

## Other Metrics

- Token Cost

- Word Error Rate (WER): evaluates the difference between a sequence generated by a system and a reference sequence

---

For more metrics see https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall?hl=pt-br
