# Emotion Detection with Generative Pre-trained Transformers — WASSA 2023

This repository contains the code and experiments behind the paper
**["Team Bias Busters at WASSA 2023 Empathy, Emotion and Personality Shared Task: Emotion Detection with Generative Pretrained Transformers"](paper_wassa2023_emotion_detection_genai.pdf)**
(Andrew Nedilko, Yi Chu — ACL 2023, WASSA workshop).

The work was done as part of **Track 3: Emotion Classification (EMO)** of the
[WASSA 2023 Shared Task on Empathy, Emotion and Personality Detection in
Interactions and Reaction to News Stories](https://codalab.lisn.upsaclay.fr/competitions/11167),
co-located with ACL 2023. The track is a multi-label, multi-class essay-level
classification task over 8 categories (7 emotions + Neutral): *Anger, Disgust,
Fear, Hope, Joy, Neutral, Sadness, Surprise*.

## Headline Result

> **Final blind test set: macro-F1 = 0.6469, micro-F1 = 0.6996.**
> **Ranked 2nd among all participating teams.**

The winning configuration was the **GPT-3 DaVinci model** fine-tuned on the shared-task data augmented with GPT-4-generated essays for under-represented classes (*Hope*, *Surprise*, *Joy*, *Fear*). Both zero-shot and few-shot prompting of ChatGPT and GPT-4 were unable to beat the team's own XGBoost baseline — fine-tuning was what closed the gap.

## Repository Map

```
.
├── README.md                                        — this file
├── paper_wassa2023_emotion_detection_genai.pdf      — published paper (ACL 2023, WASSA)
├── requirements.txt                                 — pinned dependencies (openai 0.27.x era)
├── utils.py                                         — shared helpers imported by the notebooks
├── 01_explore_dataset.ipynb                         — EDA: distributions, length, label co-occurrence
├── 02_prepare_data.ipynb                            — cleaning, multi-label binarization, splits
├── 03_baseline_xgboost_emotion_classifier.ipynb     — XGBoost baseline (Track 3 — emotion)
├── 04_baseline_xgboost_empathy_regressor.ipynb      — XGBoost baseline (empathy regressor, side track)
├── 05_gpt3_davinci_finetune_emotion.ipynb           — winning model: fine-tuned GPT-3 DaVinci
├── 06_gpt3_davinci_finetune_empathy.ipynb           — empathy as classification with fine-tuned DaVinci
├── 07_gpt35_zero_shot_baseline.ipynb                — first zero-shot ChatGPT smoke test
├── 08_gpt35_zero_shot_prompt_eng.ipynb              — zero-shot ChatGPT with iterative prompt engineering
├── 09_gpt35_few_shot_experiments.ipynb              — few-shot ChatGPT (random + cosine-similar examples)
├── 10_gpt4_zero_shot_prompt_eng.ipynb               — zero-shot GPT-4 with prompt engineering
└── 11_gpt4_few_shot_experiments.ipynb               — few-shot GPT-4 (random + cosine-similar examples)
```

Notebooks 04 and 06 cover the related **empathy prediction** track and are not
part of the paper's published results — they are included as exploratory work.

## Methodology

The notebooks follow a deliberate progression: **EDA → baseline → zero-shot →
few-shot → fine-tune.** Each step was kept to test the next research question,
with the previous step's metric as the bar to beat.

| #   | Notebook                                          | Purpose                                                                             |
| --- | ------------------------------------------------- | ----------------------------------------------------------------------------------- |
| 01  | `01_explore_dataset.ipynb`                        | Inspect train/dev/test distributions, essay lengths, label co-occurrence            |
| 02  | `02_prepare_data.ipynb`                           | Clean text (`ftfy`, regex), multi-label binarize the 8 emotion classes              |
| 03  | `03_baseline_xgboost_emotion_classifier.ipynb`    | XGBoost over character n-gram counts (1, 7); upsampling/downsampling to balance     |
| 04  | `04_baseline_xgboost_empathy_regressor.ipynb`     | XGBoost regressor for empathy track (separate task, included for completeness)      |
| 05  | `05_gpt3_davinci_finetune_emotion.ipynb`          | Fine-tune GPT-3 DaVinci on essay → emotion; logprob-based 2nd-class decision        |
| 06  | `06_gpt3_davinci_finetune_empathy.ipynb`          | Same idea but framing the empathy track as classification                           |
| 07  | `07_gpt35_zero_shot_baseline.ipynb`               | First end-to-end pass of the ChatGPT API on dev set                                 |
| 08  | `08_gpt35_zero_shot_prompt_eng.ipynb`             | Iterate prompts, system roles, follow-up questions to improve zero-shot quality     |
| 09  | `09_gpt35_few_shot_experiments.ipynb`             | Two example-selection strategies: random N essays and cosine-nearest 30 essays      |
| 10  | `10_gpt4_zero_shot_prompt_eng.ipynb`              | Same prompt-engineering protocol applied to GPT-4                                   |
| 11  | `11_gpt4_few_shot_experiments.ipynb`              | Few-shot with GPT-4 (8K context); cost rose into 3-digit USD per experiment         |

## Results

All numbers below are on the **dev set** (208 essays) and come from Table 1 of
the paper. The official competition metric is **macro-F1**; micro-F1 is shown
as a secondary signal.

| Method                                       | Macro-F1   | Micro-F1   |
| -------------------------------------------- | ---------- | ---------- |
| Baseline XGBClassifier (char n-grams)        | 0.5057     | 0.6053     |
| Improved XGBClassifier (downsample + upsample + augment) | 0.5638 | 0.6162 |
| Zero-shot ChatGPT                            | 0.4620     | 0.5720     |
| Few-shot ChatGPT (random examples)           | 0.4744     | 0.5992     |
| Few-shot ChatGPT (most-similar examples)     | 0.4237     | 0.5906     |
| Zero-shot GPT-4                              | 0.4285     | 0.5505     |
| Few-shot GPT-4 (random examples)             | 0.4657     | 0.6300     |
| Few-shot GPT-4 (most-similar examples)       | 0.4325     | 0.5940     |
| Fine-tuned DaVinci                           | 0.5811     | 0.6877     |
| **Fine-tuned DaVinci with augmented data**   | **0.5916** | **0.6800** |

On the **final blind test set** (100 essays), the winning fine-tuned model
scored **macro-F1 = 0.6469, micro-F1 = 0.6996**, placing the team **2nd
overall**.

Two findings stood out:

- **Zero/few-shot prompting could not beat a properly tuned XGBoost baseline.**
  Even GPT-4 underperformed XGBoost on macro-F1. The baseline was harder to
  beat than the team initially expected.
- **Fine-tuning the older, smaller DaVinci model closed the gap.** Access to logits
  also enabled a probability-based decision rule for the optional second
  emotion label — something prompt-only methods couldn't replicate.

## Reproducibility

The dataset is **not redistributed in this repo**. To reproduce, register on
the shared-task page and download `WASSA23_essay_level_*.tsv`:

- **CodaLab competition:** https://codalab.lisn.upsaclay.fr/competitions/11167
- Place the files under `data/` at the project root (the notebooks expect e.g.
  `data/df_train.pkl`, `data/df_dev.pkl`, `data/WASSA23_essay_level_test.tsv`).

Install dependencies with `pip install -r requirements.txt`. The notebooks were
written against the pre-1.0 OpenAI Python SDK, so the API patterns
(`openai.Completion.create`, `openai.error.RateLimitError`,
`openai.embeddings_utils.cosine_similarity`) reflect that era — see the next
section for what would change today.

Set environment variables before running anything that hits the API:

```
export OPENAI_API_KEY="sk-..."
export OPENAI_API_KEY2="sk-..."   # used by the ChatGPT-track notebooks
```

## What I'd Do Differently in 2026

Original code was written in mid-2023. Both the modeling landscape and the
OpenAI SDK have moved; here is what would change in a fresh attempt:

- **Compare across providers.** Run the same prompts through Claude (Sonnet
  4.6 / Opus 4.7) and Gemini in addition to OpenAI. On nuanced classification
  with ambiguous label boundaries, Claude often handles the "should I add a
  second category?" judgment more conservatively than GPT-4 did at the time
  (GPT-4 was over-eager to emit a 2nd label even at temperature 0).
- **Use Structured Outputs / tool-calling for the prediction format** instead
  of regex-parsing free text. The whole `verify_label`/follow-up-question
  retry layer in notebooks 08–11 disappears when the model is constrained to
  emit a JSON object like `{"primary": "Sadness", "secondary": null}`.
- **Move to the OpenAI Batch API for offline dev-set runs.** A 208-essay
  evaluation sweep is a textbook batch workload — 50% cheaper, runs
  asynchronously, and the GPT-4 cost overruns mentioned in the paper would
  not happen.
- **Use prompt caching for the few-shot prefix.** In notebook 11 every
  dev-set example was sent with the same long block of in-context examples.
  Anthropic's prompt caching (and OpenAI's input-token caching) make the
  shared prefix effectively free after the first call — a huge win on the
  most expensive notebook.
- **Replace `text-embedding-ada-002`** (used for cosine-similar example
  retrieval in nb 09 / 11) with `text-embedding-3-large` or an open-weight
  model like `bge-m3`. The "most-similar examples" strategy underperformed
  random selection in the paper; better embeddings or a hybrid
  lexical + dense retriever might flip that result.
- **Fine-tune an open-weight model.** GPT-3 DaVinci fine-tuning is
  deprecated. The same recipe today would target Llama 3 / Qwen 2.5 / a
  modern small instruction-tuned model — cheaper, fully reproducible, and
  the logits are available without paying per token.
- **Rebuild evaluation on a proper harness** (Inspect AI, lm-eval-harness,
  or even a few hundred lines of pytest) so the experiment matrix is
  declarative and rerunnable.
- **Treat label augmentation more carefully.** GPT-4-generated synthetic
  examples helped here, but in 2026 I'd combine that with
  consistency-filtering (drop synthetic essays a held-out classifier
  misclassifies) to avoid teaching the model the augmenter's biases.
