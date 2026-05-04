# LLM API Examples — Multi-Provider Emotion Classification

Self-contained examples of calling four different LLM APIs (OpenAI, Anthropic,
Google Gemini, Hugging Face) and a head-to-head comparison on the WASSA 2023
emotion-classification task.

## Layout

| Path | What it is |
|---|---|
| `classifiers/` | SDK wrappers, one file per provider, sharing a base class and prompt template. |
| `comparison.py` | Runs all four classifiers on the WASSA dev set; writes per-API classification reports and an aggregated summary as CSVs into `data/`. |
| `requirements.txt` | Pinned floors for every SDK and library used by `comparison.py` and the classifier wrappers. |
| `Gemini_APIs_pprint.ipynb`, `LangChain_*.ipynb`, `OpenAI_Dalle_*.ipynb` | Older API quickstarts - may contain stale SDK calls. Kept for context but not maintained. |

## The comparison project

`comparison.py` answers one question: **on the same labeled data and the same
prompt, how do these four providers compare?**

- **Same prompt.** The WASSA-style instruction from the notebook (`prompt_one`),
  preserved verbatim and sent as a `system` + `user` pair so each provider gets
  the canonical message format for its API.
- **Same dev set.** Loaded from `data/df_dev.pkl`, the same path the notebook
  used. The `emotion` column (which can be a list, a string, or a `"A/B"`
  slash-joined string) is reduced to a single primary label so single-label
  metrics work uniformly.
- **Same eval.** sklearn's `classification_report` over all 8 emotion labels,
  plus accuracy / macro-F1 / weighted-F1 surfaced in the aggregated summary.
- **Outputs.** `data/classification_report_<api>_<model>.csv` per provider, and
  `data/classifier_comparison_summary.csv` aggregated and sorted by macro-F1.

Each classifier conforms to the same `BaseEmotionClassifier` interface — one
method, `classify(text) -> str` — so adding a fifth provider is a single new
file and one line in `comparison.py`.

### Default models

Each classifier defaults to a cost-conscious tier so the comparison is fair
across providers. Override via constructor argument when instantiating.

| Provider | Default model | Why |
|---|---|---|
| OpenAI | `gpt-4o-mini` | Cheap, fast, capable instruction-follower. |
| Anthropic | `claude-haiku-4-5` | Anthropic's fastest Claude 4.5-tier model. |
| Google | `gemini-2.5-flash` | Flash tier — matched to the others on cost / latency. |
| Hugging Face | `meta-llama/Llama-3.1-8B-Instruct` | Open-weights baseline, routed through HF Inference. |

## Running it

```bash
pip install -r requirements.txt

export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...        # or GEMINI_API_KEY
export HF_TOKEN=...

python comparison.py --limit 30   # smoke run on first 30 dev examples
python comparison.py              # full dev set
```

Both commands expect `data/df_dev.pkl` to exist — it's the WASSA 2023 dev set
as used by the notebook.

## Notes

- **`data/df_dev.pkl` is not committed.** See instructions about how to get it [here](https://github.com/agnedil/Portfolio-Recent/tree/main/04-ACL-2023-GenAI-Publications/Paper-1-Emotion-Classification-GPT3-Fine-Tuning).
- **Llama-3.1-8B is gated.** Your HF account must have accepted the Llama 3.1
  license. If serverless inference is unavailable on your tier, pass
  `provider="together"` (or `"fireworks"`, `"replicate"`) to
  `HuggingFaceEmotionClassifier` to route through a partner provider.
- **Exponential backoff on every API call.** All four classifiers use `tenacity`
  to retry on transient errors only — rate limits (429), 5xx, request timeouts,
  connection errors. Auth failures (401), bad requests (400), and permission
  errors (403) raise immediately. Defaults: 5 attempts, random exponential wait
  capped at 30s. Tune via `RETRY_MAX_ATTEMPTS` and `RETRY_WAIT_MAX_S` in
  `classifiers/base.py`.