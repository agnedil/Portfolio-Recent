# Emotion Detection in a Code-Switching Setting (Roman Urdu + English) — WASSA 2023

This repository contains the code and experiments behind the paper
**["Generative Pretrained Transformers for Emotion Detection in a
Code-Switching Setting"](paper_wassa2023_emotion_detection_codeswitching.pdf)**
(Andrew Nedilko — ACL 2023, WASSA workshop).

The work was done as part of **Track MCEC: Multi-Class Emotion Classification**
of the [WASSA 2023 Shared Task on Multi-Label and Multi-Class Emotion
Classification on Code-Mixed Text Messages](https://codalab.lisn.upsaclay.fr/competitions/10864),
co-located with ACL 2023.

The task: given a colloquial **code-mixed Roman Urdu + English SMS message**,
predict one of 12 categories (11 emotions + neutral): *anger, anticipation,
disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust,
neutral*. The dataset is highly imbalanced (neutral dominates; pessimism and
love are rarest) and contains substantial leakage between train, dev, and
test (handled explicitly in notebook 01).

## Headline Result

> **Final blind test set: macro-F1 = 0.7038, accuracy = 0.7313.**
> **Ranked 4th among all participating teams.**

The winning configuration was a **few-shot ChatGPT (gpt-3.5-turbo) classifier**
that, for every test message, retrieved the **top-100 most cosine-similar
training examples** (using OpenAI `text-embedding-ada-002`) and presented them
as in-context examples. This approach beat:

- the team's own **XGBoost baseline** (macro-F1 0.68, accuracy 0.705)
- the organizers' **`bert-base-multilingual-cased`** baseline (macro-F1 0.7014, accuracy 0.7298)

Notably, **zero-shot ChatGPT failed completely** on this task (macro-F1 below
0.5) — the gain came entirely from few-shot in-context examples chosen by
similarity, not from prompt engineering or model size.

## Repository Map

```
.
├── README.md                                                 — this file
├── paper_wassa2023_emotion_detection_codeswitching.pdf       — published paper (ACL 2023, WASSA)
├── requirements.txt                                          — pinned dependencies (openai 0.27.x era)
├── utils.py                                                  — shared helpers imported by the notebooks
├── 01_dedup_and_data_leakage.ipynb                           — quantify train/dev/test overlap; dedup the train set
├── 02_attempted_machine_translation.ipynb                    — Urdu → English MT exploration (Googletrans, ChatGPT)
├── 03_get_openai_embeddings.ipynb                            — pre-compute OpenAI embeddings for all texts
├── 04_baseline_xgboost_12cat.ipynb                           — XGBoost baseline for the 12-category task
├── 05_baseline_xgboost_binary.ipynb                          — XGBoost baseline for emotional vs neutral
├── 06_baseline_xgboost_threeway.ipynb                        — XGBoost baseline for negative/neutral/positive sentiment
├── 07_zero_shot_12cat_experiments.ipynb                      — first end-to-end zero-shot ChatGPT pass on 12-category
├── 08_zero_shot_binary.ipynb                                 — zero-shot ChatGPT on the binary subtask
├── 09_zero_shot_threeway.ipynb                               — zero-shot ChatGPT on the three-way sentiment subtask
├── 10_few_shot_binary_v1.ipynb                               — few-shot binary, single-prompt example concatenation
├── 11_few_shot_binary_v2.ipynb                               — few-shot binary, chat-format alternating user/assistant
├── 12_few_shot_12cat_dev.ipynb                               — few-shot 12-category on the dev set
├── 13_hybrid_leaked_labels_plus_chatgpt.ipynb                — hybrid: copy leaked labels, classify the rest with ChatGPT
├── 14_final_baseline_with_hp_tuning.ipynb                    — final XGBoost baseline with HP tuning + leakage exploit
└── 15_winning_few_shot_12cat_test.ipynb                      — winning submission: few-shot 12-category on test set
```

## Methodology

The narrative arc was: **understand the data → build classical baselines per
task variant → confirm zero-shot ChatGPT couldn't carry the task → switch to
few-shot with intelligent example selection.** The binary and three-way
classifiers (notebooks 05, 06, 08, 09) were built as *debugging surrogates* —
simpler subtasks where it was easier to see what ChatGPT was doing wrong before
returning to the full 12-category problem.

| #   | Notebook                                          | Purpose                                                                                  |
| --- | ------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 01  | `01_dedup_and_data_leakage.ipynb`                 | Quantify overlap between splits (~50% of dev was in train); deduplicate the training set |
| 02  | `02_attempted_machine_translation.ipynb`          | Try Urdu → English MT (Googletrans, ChatGPT); decide it added noise and dropped         |
| 03  | `03_get_openai_embeddings.ipynb`                  | Pre-compute `text-embedding-ada-002` embeddings for all messages (used by nb 12, 15)    |
| 04  | `04_baseline_xgboost_12cat.ipynb`                 | XGBoost over character n-grams (1, 5) — main task baseline (macro-F1 below 0.6)         |
| 05  | `05_baseline_xgboost_binary.ipynb`                | Same recipe for emotional vs neutral subtask                                            |
| 06  | `06_baseline_xgboost_threeway.ipynb`              | Same recipe for negative/neutral/positive sentiment subtask                             |
| 07  | `07_zero_shot_12cat_experiments.ipynb`            | First zero-shot pass; iterate on prompts ("translate then classify", direct, etc.)      |
| 08  | `08_zero_shot_binary.ipynb`                       | Zero-shot binary; surfaced the model's neutral-bias                                     |
| 09  | `09_zero_shot_threeway.ipynb`                     | Zero-shot three-way to isolate sentiment signal                                         |
| 10  | `10_few_shot_binary_v1.ipynb`                     | Few-shot v1: concatenate ~135 examples per user message into a single prompt            |
| 11  | `11_few_shot_binary_v2.ipynb`                     | Few-shot v2: alternating user/assistant chat-format examples (more token-efficient)     |
| 12  | `12_few_shot_12cat_dev.ipynb`                     | Few-shot 12-category on dev set: cosine-similar top-100 examples per message            |
| 13  | `13_hybrid_leaked_labels_plus_chatgpt.ipynb`      | Test-set hybrid: copy leaked labels for matched texts, ChatGPT-classify the rest        |
| 14  | `14_final_baseline_with_hp_tuning.ipynb`          | Final XGBoost baseline with hyperparameter tuning (the bar to beat)                     |
| 15  | `15_winning_few_shot_12cat_test.ipynb`            | **Winning submission** — few-shot 12-category on the blind test set                     |

The key insight (paper §4.2): with a 4096-token context window and a ~110k-token
training set, you cannot fit all examples in a prompt. Selecting which ~100
examples to show via cosine similarity to the query message is what unlocked
the result.

## Results

The official competition metric is **macro-F1**; **accuracy** is the secondary
metric. All numbers below are on the **final blind test set** and come from
Table 1 of the paper.

| Method                                                | Macro-F1   | Accuracy   |
| ----------------------------------------------------- | ---------- | ---------- |
| Baseline XGBClassifier (char n-grams, this team)      | 0.6800     | 0.7050     |
| Baseline `bert-base-multilingual-cased` (organizers)  | 0.7014     | 0.7298     |
| **Few-shot ChatGPT (most-similar examples)** — winning | **0.7038** | **0.7313** |

Two findings from the dev-set experimentation that are worth noting:

- **Zero-shot was unusable.** Across multiple prompt designs the macro-F1
  stayed below 0.5. The 12-category label space is too fine-grained for
  zero-shot, and the model defaulted to "neutral" too often.
- **Random-chunk few-shot beat zero-shot but lost to similarity-selected
  few-shot.** Showing 100 *random* training examples per message helped, but
  showing the 100 *most-similar* ones (per cosine on OpenAI embeddings) was
  what closed the gap to the BERT baseline — and then beat it.

## Reproducibility

The dataset is **not redistributed in this repo**. To reproduce, register on
the shared-task page and download the MCEC train/dev/test files:

- **CodaLab competition:** https://codalab.lisn.upsaclay.fr/competitions/10864
- Place files under `data/` at the project root. The notebooks expect
  `data/mcec_train.csv`, `data/mcec_dev.csv`, `data/mcec_test.csv`,
  `data/sample_submission/predictions_MCEC.csv`, and pickled translated /
  embedded variants produced by notebooks 02, 03 (e.g.
  `data/mcec_train_translated.pkl`, `data/df_dev_100_closest_GptEmbeddings.pkl`).

Install dependencies with `pip install -r requirements.txt`. The notebooks were
written against the pre-1.0 OpenAI Python SDK, so the API patterns
(`openai.ChatCompletion.create`, `openai.error.RateLimitError`,
`openai.embeddings_utils.cosine_similarity`) reflect that era.

Set the OpenAI API key before running anything that hits the API:

```
export OPENAI_API_KEY="sk-..."
```

## What I'd Do Differently in 2026

This code was written in April–May 2023. Both the modeling landscape and the
OpenAI SDK have moved; here is what would change in a fresh attempt:

- **Compare across providers.** Run the same retrieval-augmented few-shot
  pipeline through Claude (Sonnet 4.6 / Opus 4.7) and Gemini in addition to
  OpenAI. Presumably, Claude's stronger multilingual handling — particularly for low-
  resource languages like Urdu — could plausibly raise the macro-F1 ceiling
  on this task without any other changes.
- **Use the OpenAI Batch API for the dev sweep.** A dev-set evaluation over
  ~1,200 messages with 100 in-context examples each is a textbook batch
  workload — 50% cheaper, runs asynchronously, no rate-limit choreography.
- **Use Structured Outputs / tool-calling to constrain the response.** The
  `verify_label` / followup-question retry layer in notebooks 07–15 only
  exists because the model emitted free text. With a JSON-schema-constrained
  output like `{"category": "<one of N>"}`, the entire post-processing layer
  collapses.
- **Cache the in-context prefix.** With 100 examples per message and ~1,200
  test messages, the same example block (per query) is sent once. But for
  the *random-chunk* variants in notebooks 10, 11, the same chunk is sent to
  many queries — Anthropic prompt caching / OpenAI input-token caching makes
  the shared prefix effectively free after the first call.
- **Replace `text-embedding-ada-002`.** The 2023-era ada-002 was the only
  reasonable choice at the time. In 2026 it is dominated by
  `text-embedding-3-large` or open-weight retrievers like `bge-m3` (which
  also handles low-resource and code-mixed languages better). The
  similarity-based example selection — the single most important component
  of the winning system — would benefit directly.
- **Treat data leakage as a first-class signal, not a problem to remove.**
  About half of the dev set was duplicated in the training set. Notebook 01
  removes leakage to enable honest evaluation, but the leaked labels were
  still useful at *prediction* time (notebook 13 exploits this for the test
  set). A 2026 pipeline would build retrieval-with-exact-match as a fast
  first hop, falling through to the LLM only on novel messages.
- **Try fine-tuning an open-weight multilingual model** (e.g. Llama 3
  multilingual, Qwen 2.5, or AfroLid-style smaller multilingual models) for
  cost and reproducibility. With a ~6k deduplicated training set, supervised
  fine-tuning is well within reach and the model could be checkpointed and
  shared without API dependence.
- **Build a proper eval harness** (Inspect AI, lm-eval-harness, or even a
  small pytest suite) so the dozens of dev-set experiments become a
  declarative grid.
