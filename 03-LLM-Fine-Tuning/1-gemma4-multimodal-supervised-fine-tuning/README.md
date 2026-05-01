# Gemma-4 Fine-Tuning Scripts (Unsloth + LoRA)

Production-grade Python conversions of the Unsloth Gemma-4 notebooks. Each
script loads a Gemma-4 model, attaches LoRA adapters, runs a fine-tuning or
reinforcement-learning loop, and saves the resulting adapters.

| Script | Modality | Algorithm | Default base model |
| --- | --- | --- | --- |
| `gemma4-31b-text-fine-tune.py` | Text | SFT (supervised) | `unsloth/gemma-4-31B-it` |
| `gemma4-31b-vision-fine-tune.py` | Image + text | SFT (supervised) | `unsloth/gemma-4-E4B-it` |
| `gemma4-31b-audio-fine-tune.py` | Audio + text (ASR) | SFT (supervised) | `unsloth/gemma-4-E4B-it` |
| `gemma4-31b-grpo-reinforcement-learning-fin-tune.py` | Text (Sudoku) | GRPO (RL) | `unsloth/gemma-4-E2B-it` |

---

## 1. Environment setup

A CUDA-capable GPU is required. The notebooks were validated on Tesla T4 / L4 /
A100, but any modern NVIDIA GPU with sufficient VRAM works. The 31B model needs
two GPUs (the script defaults to `device_map="balanced"`); the E2B/E4B models
fit on a single 14–24 GB card.

```bash
# Create and activate a virtual environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate

# Install the unified dependency set
pip install -r requirements.txt
```

If you run inside Google Colab, follow the Unsloth Colab installation flow
instead — it pins `xformers` against the active `torch` build.

For gated Hugging Face models, export a token (or pass `--hf-token`):

```bash
export HF_TOKEN=hf_xxx
```

---

## 2. Running the scripts

All four scripts share the same CLI shape: every hyperparameter has a default
that matches the source notebook, so `python <script>.py` reproduces the
reference run. Override any value via flags. Use `--help` on any script to see
the full list.

### 2.1 Text SFT

```bash
# Reproduce the notebook (60 steps, FineTome-100k subset)
python gemma4-31b-text-fine-tune.py

# Full epoch, smaller model, custom output path
python gemma4-31b-text-fine-tune.py \
    --model-name unsloth/gemma-4-E4B-it \
    --max-steps -1 --num-train-epochs 1 \
    --output-dir ./adapters/text-finetome
```

### 2.2 Vision SFT

```bash
# LaTeX OCR fine-tune at 60 steps (notebook default)
python gemma4-31b-vision-fine-tune.py

# Use a different vision dataset and instruction
python gemma4-31b-vision-fine-tune.py \
    --dataset-name unsloth/Radiology_mini \
    --instruction "Describe this radiograph." \
    --max-steps 200
```

### 2.3 Audio SFT

```bash
# German ASR on Emilia-DE, 3000 samples, 60 steps
python gemma4-31b-audio-fine-tune.py

# Train on the entire split, larger batch, custom prompts
python gemma4-31b-audio-fine-tune.py \
    --num-samples -1 \
    --per-device-train-batch-size 4 \
    --system-prompt "Transcribe the following speech in English." \
    --user-instruction "Transcribe."
```

### 2.4 GRPO reinforcement learning

```bash
# Sudoku-solving RL with the notebook defaults (E2B-it, 60 GRPO steps)
python gemma4-31b-grpo-reinforcement-learning-fin-tune.py

# Easier puzzles, more rollouts per step, longer run
python gemma4-31b-grpo-reinforcement-learning-fin-tune.py \
    --sudoku-difficulty 30 \
    --num-generations 4 \
    --max-steps 300
```

---

## 3. What happens at runtime

### 3.1 Common pipeline (all four scripts)

1. **Argument parsing** — `argparse` reads paths, model loading flags, LoRA
   knobs, and training hyperparameters; sensible defaults come straight from
   the source notebook.
2. **Base model load** — `FastModel` / `FastVisionModel` from Unsloth pulls
   the requested Gemma-4 checkpoint, optionally in 4-bit quantization, and
   patches it for fast training.
3. **LoRA attachment** — `get_peft_model` installs LoRA adapters so only ~0.2–
   1% of parameters are trainable; the base weights stay frozen.
4. **Dataset prep** — a Hugging Face dataset is loaded, normalized into a chat
   format Gemma-4 understands, and wired into the trainer.
5. **Training** — TRL's `SFTTrainer` (or `GRPOTrainer` for RL) iterates over
   the dataset; peak GPU memory is logged before/after.
6. **Save** — the LoRA adapters and the tokenizer/processor are written to
   `--output-dir`. The base model is *not* saved; load it together with the
   adapters via `FastModel.from_pretrained("<output-dir>")` for inference.

### 3.2 Per-script differences

**`gemma4-31b-text-fine-tune.py`**
Loads `unsloth/gemma-4-31B-it` in 4-bit on two GPUs (`device_map="balanced"`)
and applies the `gemma-4-thinking` chat template. The dataset
(`mlabonne/FineTome-100k`, first 3000 rows) is rendered into the Gemma-4
multi-turn template; a `<bos>` prefix is stripped so the processor adds only
one. `train_on_responses_only` masks the user prompt so the loss is computed
on assistant turns only. LoRA: `r=8`, `alpha=8`, language layers + attention +
MLP.

**`gemma4-31b-vision-fine-tune.py`**
Loads `unsloth/gemma-4-E4B-it` via `FastVisionModel` with Unsloth-style
gradient checkpointing. Each row of `unsloth/LaTeX_OCR` becomes a 2-turn chat
(user: instruction + image, assistant: LaTeX). Training runs through
`UnslothVisionDataCollator` with the vision-specific SFTConfig flags
(`remove_unused_columns=False`, `dataset_kwargs={"skip_prepare_dataset": True}`,
`max_length=2048`). LoRA: `r=32`, `alpha=32`, `target_modules="all-linear"`,
vision + language layers both trained.

**`gemma4-31b-audio-fine-tune.py`**
Loads `unsloth/gemma-4-E4B-it` via `FastModel`. The `kadirnar/Emilia-DE-B000000`
dataset is sliced to 3000 samples and resampled to 16 kHz; each sample becomes
a 3-turn chat (system: "transcribe accurately", user: audio + instruction,
assistant: ground-truth transcript). The same `UnslothVisionDataCollator`
handles audio token alignment. LoRA: `r=8`, `alpha=16`, with target modules
extended to include Gemma-4's audio submodules (`post`, `linear_start`,
`linear_end`, `embedding_projection`, `ffw_layer_1`, `ffw_layer_2`,
`output_proj`); vision layers are disabled.

**`gemma4-31b-grpo-reinforcement-learning-fin-tune.py`**
Loads `unsloth/gemma-4-E2B-it` in 16-bit LoRA mode (no 4-bit quantization).
Instead of a static labeled dataset, the script defines a self-contained
`SudokuGame` environment plus an `execute_strategy` runner with a 10-second
timeout. The "dataset" is 1000 copies of one prompt asking the model to write
a `def strategy(board, initial)` Python function. Three reward functions score
each generation:

- `function_works` — does the snippet parse and lock down without error?
- `no_cheating` — does it avoid third-party imports? (heavy penalty if not)
- `strategy_succeeds` — when executed against a freshly generated puzzle, how
  many valid moves does the strategy land before failing? Solving the puzzle
  awards +30; valid moves earn +0.2 each.

`GRPOTrainer` from TRL uses these rewards to update only the LoRA adapters via
group-relative policy optimization; the base Gemma weights stay frozen. The
training metric to watch is the `reward` column — it should trend upward over
the run.

---

## 4. Outputs

Each script writes to `--output-dir` (default `gemma_4_lora`):

```
gemma_4_lora/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json / processor_config.json / chat_template.jinja
└── ...
```

To run inference with a saved adapter:

```python
from unsloth import FastModel  # or FastVisionModel for vision/RL adapters
model, tokenizer = FastModel.from_pretrained("gemma_4_lora", load_in_4bit=True)
```

For 16-bit merged checkpoints or GGUF exports, see the Unsloth documentation —
the saving cells are intentionally not included in these scripts so the
default behavior is a small, fast adapter dump.
