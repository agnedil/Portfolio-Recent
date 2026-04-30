# Llama 3 8B Fine-Tuning (HF Trainer)

Production-grade fine-tuning.

| Script | Purpose |
| --- | --- |
| `train.py` | Fine-tune Llama 3.1 8B with the Hugging Face `Trainer` |
| `inference.py` | Generate text from a fine-tuned checkpoint via the `text-generation` pipeline |
| `serve.py` | Expose the fine-tuned model behind a FastAPI `/generate` endpoint |
| `Dockerfile` | Container image for `serve.py` (the article's deployment recipe) |

> **Model size:** the scripts default to `meta-llama/Meta-Llama-3.1-8B-Instruct`.
> Full fine-tuning of an 8B model needs substantial VRAM (~80 GB+). Use
> `--freeze-base` (the article's recommendation for limited memory) or pass
> a smaller variant via `--model-name` if you don't have a multi-GPU box.

Note: You can use the script to fine-tune other 8B alternatives without code changes: NousResearch/Meta-Llama-3.1-8B-Instruct
(un-gated mirror), unsloth/llama-3.1-8b-Instruct (faster loads), or any third-party Llama-architecture 8B like mistralai/Ministral-8B-Instruct-2410.

---

## 1. Environment setup

```bash
# Create and activate a fresh environment (Python 3.10+ recommended)
conda create -n llama_finetune python=3.10 -y
conda activate llama_finetune

# Install dependencies
pip install -r requirements.txt

# One-time auth (Llama 3 is gated on Hugging Face)
huggingface-cli login

# Optional: configure Accelerate / Weights & Biases
accelerate config
wandb login
```

GPU sanity check (mirrors the article's snippet):

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## 2. Training (`train.py`)

Reproduce the article's defaults (3 epochs, batch 4 × grad-accum 8, lr 5e-5,
fp16 on, save every 1000 steps, log every 500):

```bash
python train.py
```

Common overrides:

```bash
# Custom JSONL with input/output pairs (article's preprocessing format)
python train.py --jsonl-path ./data/train.jsonl

# Single-epoch run + Weights & Biases tracking
python train.py \
    --num-train-epochs 1 \
    --wandb-project llama-finetune \
    --wandb-run-name experiment_1

# Limited-memory training: freeze the base, only train the LM head
python train.py --freeze-base

# Apply text cleaning + synonym augmentation before tokenization
python train.py --clean-text --augment

# Class-imbalance handling: oversample a minority label and use weighted loss
python train.py \
    --oversample-label minority --oversample-target-size 2000 \
    --class-weights 0.7 1.3
```

Run `python train.py --help` for the full flag list.

---

## 3. Inference (`inference.py`)

```bash
# Default prompt from the article
python inference.py --model-path ./llama_finetuned

# Custom prompt + multiple completions
python inference.py \
    --model-path ./llama_finetuned \
    --prompt "Summarize the second law of thermodynamics." \
    --max-length 200 --num-return-sequences 3
```

---

## 4. Serving (`serve.py`)

Local launch:

```bash
LLAMA_MODEL_PATH=./llama_finetuned uvicorn serve:app --host 0.0.0.0 --port 8000
```

Container launch:

```bash
docker build -t llama-finetuned-serve .
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/llama_finetuned:/app/llama_finetuned \
    -e LLAMA_MODEL_PATH=/app/llama_finetuned \
    llama-finetuned-serve
```

Call the endpoint:

```bash
curl -X POST "http://localhost:8000/generate?prompt=What+is+a+transformer%3F"
```

---

## 5. What happens at runtime

### 5.1 Common pipeline

All scripts share two pieces of setup: load the Llama 3 8B tokenizer +
weights from Hugging Face, and read configuration via `argparse` / environment
variables. After that, each script does its specific job (training,
generation, or serving).

### 5.2 Per-script behavior

**`train.py`**
1. **GPU validation** — logs CUDA availability and device name.
2. **Model + tokenizer load** — `LlamaForCausalLM.from_pretrained` and
   `AutoTokenizer.from_pretrained`; pads with EOS if the tokenizer has no
   pad token.
3. **Optional layer freezing** — when `--freeze-base` is set, every parameter
   in `model.base_model` has `requires_grad = False`, so only the LM head
   updates (the article's recipe for tight VRAM).
4. **Dataset prep** — loads either a custom JSONL (`--jsonl-path`) or an HF
   dataset (`--dataset-name`, default `squad`). Optional steps from the
   article: regex `clean_text` (lowercase + strip non-alphanumerics),
   `nlpaug` synonym augmentation, and `sklearn.utils.resample`-based
   oversampling for an imbalanced label.
5. **Tokenization** — `tokenizer(...)` with truncation/padding to
   `--max-length`; `labels` mirror `input_ids` for causal LM training. The
   data is split into train / eval (`--eval-split-ratio`).
6. **Training arguments** — matches the article exactly: batch size 4,
   gradient accumulation 8, 3 epochs, lr 5e-5, save every 1000, log every
   500, save_total_limit 2, fp16 on by default. `report_to=["wandb"]` is
   added when `--wandb-project` is set.
7. **Trainer** — standard `Trainer` unless `--class-weights` is provided, in
   which case a `WeightedLossTrainer` subclass applies a weighted
   `CrossEntropyLoss` (article's class-imbalance recipe).
8. **Save** — `model.save_pretrained` + `tokenizer.save_pretrained` write
   the fine-tuned weights to `--output-dir` (default `./llama_finetuned`).

**`inference.py`**
Loads the checkpoint with the HF `text-generation` pipeline and prints
completions for `--prompt`. Identical to the article's evaluation snippet
but parameterized via CLI flags (max length, number of sequences, device).

**`serve.py`**
A minimal FastAPI app that loads the checkpoint at startup (so the first
request is fast) and exposes a single `POST /generate` endpoint which
tokenizes the prompt, calls `model.generate`, and returns the decoded text.
Configurable via `LLAMA_MODEL_PATH` and `LLAMA_MAX_LENGTH` env vars.

**`Dockerfile`**
`python:3.10-slim` base, installs FastAPI + uvicorn + transformers + torch,
copies the project, and runs `uvicorn serve:app` on port 8000. Copied from
the article verbatim with the addition of `torch` (required by transformers
at import time) and the working CMD pointing at `serve:app`.

---

## 6. Outputs

After training, `--output-dir` contains the standard HF artifacts:

```
llama_finetuned/
├── config.json
├── generation_config.json
├── model.safetensors (or sharded model-*.safetensors)
├── tokenizer.json / tokenizer_config.json / special_tokens_map.json
└── ...
```

Reload it anywhere with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./llama_finetuned")
tokenizer = AutoTokenizer.from_pretrained("./llama_finetuned")
```
