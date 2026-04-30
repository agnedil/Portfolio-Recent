# Preference Optimization (DPO / ORPO / KTO)

Offline preference-based alignment using TRL. A single `train.py` dispatches
to the right trainer based on `--method {dpo,orpo,kto}`. All three are
*alignment* methods — they teach the model to prefer one response over
another — but they bypass the heavy machinery of online RL (no rollouts,
no reward model, no PPO loop).

| File | Purpose |
| --- | --- |
| `train.py` | Fine-tune via DPO, ORPO, or KTO (TRL trainers) |
| `inference.py` | Generate text from a preference-tuned checkpoint |

---

## 1. The three variants

**DPO — Direct Preference Optimization**
Contrastive loss on `(prompt, chosen, rejected)` triples. The model is pushed
to assign higher likelihood to `chosen` and lower likelihood to `rejected`,
*relative to a frozen reference model*. With LoRA, the reference is the
unmodified base — no second copy in memory.

**ORPO — Odds-Ratio Preference Optimization**
Same `(chosen, rejected)` data as DPO, but combines an SFT loss on the
chosen response with an odds-ratio penalty on the rejected one. **No
reference model needed.** Often produces good results in a single training
stage instead of SFT-then-DPO.

**KTO — Kahneman-Tversky Optimization**
Unary preferences: each row is `(prompt, completion, label)` where `label`
is a thumbs-up/down bool. Useful when you only have one-sided feedback —
e.g. user clicks, ratings, or moderation flags — without paired
alternatives.

### Choosing between them

| If you have... | Use |
| --- | --- |
| Pairs of (chosen, rejected) responses | **DPO** (most common, well-tested) |
| Pairs but want a single-stage recipe (no separate SFT) | **ORPO** |
| Only one-sided thumbs-up / thumbs-down labels | **KTO** |

### How does this differ from the other fine-tuning methods in this repo?

| Method | Signal | Data shape | Online rollouts? | Reward model? |
| --- | --- | --- | --- | --- |
| **SFT** (text/vision/audio scripts) | Ground-truth tokens | `(prompt, response)` | No | No |
| **GRPO RL** (Sudoku script) | Programmatic reward fns | Prompts only; outputs scored online | Yes | No |
| **Continued pretraining** | Raw text | Plain corpus | No | No |
| **Distillation** | Teacher logits | `(prompt, response)` + teacher | No | No (teacher) |
| **DPO / ORPO** | Preference *between* responses | `(prompt, chosen, rejected)` | No | No |
| **KTO** | Preference per response | `(prompt, completion, label)` | No | No |

DPO/ORPO/KTO sit between SFT and RL: the data is richer than SFT (it carries
preference information) but the training loop is as simple as SFT (no
rollouts).

---

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # Llama models are gated
```

Default base model is `meta-llama/Llama-3.2-1B-Instruct` so the demo runs on
a single mid-range GPU with LoRA. Override with `--model-name` for any
causal-LM-architecture model.

---

## 3. Running

### DPO (default)

```bash
# Default: Llama-3.2-1B-Instruct + LoRA on UltraFeedback-binarized
python train.py

# Larger model, more samples
python train.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --max-train-samples 5000
```

### ORPO

```bash
python train.py --method orpo
```

Reuses the DPO dataset (`trl-lib/ultrafeedback_binarized`) since the data
shape is the same.

### KTO

```bash
python train.py --method kto
```

Auto-switches to `trl-lib/kto-mix-14k` (unary `label` format). Override with
`--dataset-name` if you have your own thumbs-up/down dataset.

### Inference

```bash
# Load a fully-saved model
python inference.py --model-path ./preference_finetuned \
    --prompt "Explain why entropy increases in closed systems."

# Load LoRA adapters on top of the base model
python inference.py \
    --base-model meta-llama/Llama-3.2-1B-Instruct \
    --model-path ./preference_finetuned
```

---

## 4. What happens at runtime

### 4.1 Common pipeline

1. **Argument parsing** — the `--method` flag selects DPO / ORPO / KTO; all
   other flags (paths, LoRA, batch sizes, schedule) are shared.
2. **Tokenizer + base model load** — `AutoTokenizer` and
   `AutoModelForCausalLM`; pad token falls back to EOS if missing.
3. **Dataset load** — defaults to a method-appropriate TRL dataset, or
   whatever you pass via `--dataset-name`.
4. **LoRA** — when `--use-lora` (default), a `LoraConfig` is built and
   passed to the TRL trainer, which wraps the base model with PEFT
   adapters. The frozen base doubles as the DPO/KTO reference (no second
   copy needed).
5. **Trainer dispatch** — `--method` chooses `DPOTrainer`, `ORPOTrainer`,
   or `KTOTrainer`. The corresponding `*Config` (DPO/ORPO/KTO) is built
   from shared training kwargs plus method-specific ones (`beta`,
   `orpo_beta`).
6. **Train + save** — `trainer.train()` runs the loop;
   `trainer.save_model()` writes adapters (if LoRA) or the full model.

### 4.2 Per-method differences inside the trainer

- **DPO** — needs both a *policy* (the trainable model) and a *reference*.
  With PEFT, TRL constructs the reference on-the-fly by disabling the
  adapters during the reference forward pass. Loss is the log-ratio
  contrast between policy and reference on chosen vs. rejected.
- **ORPO** — single forward pass per sample; the loss combines an SFT-style
  cross-entropy on the chosen response with an odds-ratio penalty on the
  rejected one. No reference model.
- **KTO** — applies a Kahneman-Tversky utility shape to the policy /
  reference log-ratio per sample. Each example contributes a positive or
  negative term based on its `label` bool.

---

## 5. Outputs

`--output-dir` (default `./preference_finetuned`) contains the saved
adapters or merged model:

```
preference_finetuned/
├── adapter_config.json          # if LoRA
├── adapter_model.safetensors    # if LoRA
├── tokenizer.json / tokenizer_config.json
└── ...
```

Reload via `AutoModelForCausalLM.from_pretrained(...)` (or `PeftModel` for
adapter-only checkpoints — see `inference.py`).
