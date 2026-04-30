# Continued Pre-Training (CPT / DAPT)

Take a pre-trained base model and run *more* unsupervised next-token
prediction on a domain corpus to make it "smarter" about that domain. Same
loss the base model was originally trained with — applied to your data.
Often used as a first stage before SFT for niche domains (legal, medical,
code, biology).

| File | Purpose |
| --- | --- |
| `train.py` | Run continued pre-training on a text dataset |
| `inference.py` | Generate text from the domain-adapted model |

---

## 1. What is CPT?

Continued pre-training (a.k.a. domain-adaptive pre-training, DAPT) is the
*same* causal-LM objective as the original pre-training run: tokenize a
corpus, group tokens into fixed-length blocks, predict the next token. The
only difference is the data — your domain corpus instead of the broad web.

The model that comes out is a base LM with **better priors for your
domain**: it knows the jargon, citation styles, code idioms, etc. It is
**not instruction-tuned** unless you started from an instruct checkpoint.
Most teams do `Base → CPT → SFT (→ DPO)` to get a domain-aligned
chatbot.

### How does this differ from the other fine-tuning methods in this repo?

| Method | Data shape | Loss | Produces a chatbot? |
| --- | --- | --- | --- |
| **CPT** | Plain text corpus | Causal LM next-token | No — base model with better domain priors |
| **SFT** (text/vision/audio) | `(prompt, response)` | Cross-entropy on response tokens | Yes |
| **GRPO RL** | Prompts only; reward fns | Policy gradient | Yes (aligned) |
| **DPO / ORPO / KTO** | Preferences | Preference-contrast loss | Yes (aligned) |
| **Distillation** | Teacher outputs / logits | KL to teacher | Yes (mimics teacher) |

CPT is the *least task-specific* of the bunch. There's no chat template, no
instruction format, no labels file — just a corpus.

---

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # Llama models are gated
```

Default base model is `meta-llama/Llama-3.2-1B` (the *base*, non-instruct
variant). For task-following behavior afterwards, plan to run an SFT pass.

---

## 3. Running

### Default demo (wikitext-2)

```bash
python train.py
```

Loads `wikitext-2-raw-v1`, tokenizes, groups into 1024-token blocks, and
runs 1 epoch.

### Custom domain corpus

```bash
# Any HF dataset with a text field
python train.py \
    --dataset-name allenai/c4 --dataset-config en \
    --max-train-samples 50000 \
    --block-size 2048 \
    --num-train-epochs 1

# Local files
python train.py \
    --dataset-name json --dataset-config "" \
    --train-split train \
    --text-field content
```

For local files, point `--dataset-name` at `text` / `json` / `csv` and pass
`--data_files` via the `datasets` cache directly, or pre-load into a HF
hub dataset.

### Larger base model

```bash
python train.py --model-name meta-llama/Llama-3.1-8B
```

(8B base + full fine-tuning needs significant VRAM. Combine with FSDP /
DeepSpeed via `torchrun` if needed — see the `multi-gpu/` folder.)

### Inference

```bash
python inference.py --model-path ./cpt_finetuned \
    --prompt "Recent advances in superconductors include"
```

---

## 4. What happens at runtime

1. **Argument parsing** — paths, dataset, blocking, training schedule.
2. **Model + tokenizer load** — `AutoModelForCausalLM.from_pretrained` and
   `AutoTokenizer.from_pretrained`. Pad token falls back to EOS.
3. **Dataset load** — `load_dataset(args.dataset_name, args.dataset_config)`,
   then `train_split` (and optionally `eval_split`).
4. **Tokenization** — straight `tokenizer(text)` with no padding/truncation.
   Each row becomes a list of token ids.
5. **Blocking** — `group_texts` concatenates all token ids, drops the
   trailing remainder, and slices into `--block-size` chunks. This is the
   standard LM-training trick: no padding waste, every batch is full-length.
6. **Labels** — `labels = input_ids.copy()`. Causal LM training: the model
   predicts each position from the previous ones, automatically shifted by
   the model itself.
7. **Trainer** — standard HF `Trainer` with
   `DataCollatorForLanguageModeling(mlm=False)`. Schedule defaults to
   cosine LR, lower than typical SFT (5e-5) so the model doesn't drift far
   from base.
8. **Train + evaluate + save** — `Trainer.train()`, optional
   `Trainer.evaluate()` (eval loss / perplexity), `trainer.save_model()`.

---

## 5. Outputs

`--output-dir` (default `./cpt_finetuned`) gets the standard HF artifacts:

```
cpt_finetuned/
├── config.json
├── model.safetensors (or sharded)
├── tokenizer.json / tokenizer_config.json
└── ...
```

Reload anywhere with `AutoModelForCausalLM.from_pretrained(...)`. The output
is a base LM — feed it into your SFT pipeline next if you want a chatbot.
