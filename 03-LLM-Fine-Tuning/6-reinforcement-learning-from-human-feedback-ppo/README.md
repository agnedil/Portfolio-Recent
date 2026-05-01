# RLHF with PPO (Clip + Adaptive-KL)

Reinforcement Learning from Human Feedback using Proximal Policy
Optimization. A trainable **policy** generates responses, a frozen
**reward model** scores them, and PPO updates the policy to maximize
reward while staying close to a frozen **reference** copy of the policy.

A single `train.py` runs both classic PPO variants — switch with
`--method {clip,kl}`.

| File | Purpose |
| --- | --- |
| `train.py` | RLHF training loop with PPO-Clip or adaptive-KL PPO (uses TRL's `PPOTrainer`) |
| `train_from_scratch.py` | Same algorithm + CLI in pure PyTorch — value head, GAE, PPO loss, adaptive-KL controller, and training loop all implemented manually (no TRL) |
| `inference.py` | Generate text from a PPO-tuned policy |

---

## 1. PPO variants in one minute

Both come from Schulman et al. (2017) and both penalize deviation from the
old policy — they just do it differently:

- **PPO-Clip (`--method clip`)** — the dominant variant in practice.
  Clips the importance-weighted ratio so any update step that pushes the
  ratio outside `[1-ε, 1+ε]` gets a flat (zero-gradient) penalty. A small
  fixed KL term is kept for stability but isn't the main constraint.
- **Adaptive-KL PPO (`--method kl`)** — clipping is effectively disabled
  (very large `cliprange`); a KL-divergence penalty between the current
  policy and the old policy *is* the main constraint, and its coefficient
  is **adapted** after each step to keep KL near `--target-kl`.

Practically: PPO-Clip is more popular because it's simpler and tends to be
more stable. Adaptive-KL is useful when you have a hard constraint on how
far the policy may drift per step.

### How does this differ from the other fine-tuning methods in this repo?

| Method | Online rollouts? | Reward model? | Reference model? | Data shape |
| --- | --- | --- | --- | --- |
| **PPO RLHF** | Yes | Yes (separate classifier) | Yes (frozen policy copy) | Prompts only |
| **GRPO RL** (Gemma Sudoku) | Yes | No (programmatic reward fns) | Implicit via KL | Prompts only |
| **DPO / ORPO / KTO** | No | No | DPO/KTO yes; ORPO no | Preference pairs / unary labels |
| **SFT** | No | No | No | (prompt, response) pairs |
| **CPT** | No | No | No | Raw text |
| **Distillation** | No | No (uses a teacher LM) | No | (prompt, response) + teacher |

PPO is the heaviest (three models in VRAM, online generation per step) but
also the most expressive — it can optimize *any* reward function the reward
model represents.

---

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # Llama models are gated
```

Memory budget is the main constraint. Defaults
(`Llama-3.2-1B-Instruct` policy + DeBERTa reward model + LoRA + bf16) fit
on a single mid-range GPU. For larger policies, enable quantization:

```bash
# 4-bit policy via bitsandbytes NF4
python train.py --load-in-4bit

# Or 8-bit
python train.py --load-in-8bit
```

When LoRA is active, the **frozen reference** is constructed by disabling
adapters on the same base model — no second copy in VRAM. This is the
standard PEFT trick that makes RLHF tractable on a single GPU.

---

## 3. Running

### PPO-Clip (default)

```bash
# Default: Llama-3.2-1B-Instruct + LoRA on Alpaca prompts,
# scored by the OpenAssistant reward model
python train.py

# Larger policy with 4-bit quantization, longer run
python train.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --load-in-4bit \
    --max-steps 500 --batch-size 16
```

### Adaptive-KL PPO

```bash
python train.py --method kl --target-kl 6.0
```

Tighter constraint on policy drift:

```bash
python train.py --method kl --target-kl 2.0 --init-kl-coef 0.2
```

### Custom reward model / prompt dataset

```bash
python train.py \
    --reward-model your-org/custom-reward-model \
    --dataset-name HuggingFaceH4/ultrafeedback_binarized \
    --prompt-field prompt
```

### Inference

```bash
# Saved as a full model
python inference.py --model-path ./ppo_finetuned \
    --prompt "Summarize the second law of thermodynamics."

# Saved as bare LoRA adapters (default when --use-lora is on)
python inference.py \
    --base-model meta-llama/Llama-3.2-1B-Instruct \
    --model-path ./ppo_finetuned
```

---

## 4. What happens at runtime

### 4.1 Setup phase

1. **Argument parsing** — `--method` chooses Clip vs KL; quantization /
   LoRA / dataset / reward model knobs are all CLI-driven.
2. **Quantization config** — if `--load-in-4bit` or `--load-in-8bit`, a
   `BitsAndBytesConfig` is built (NF4 + double-quant for 4-bit by default).
3. **Policy load** — `AutoModelForCausalLMWithValueHead.from_pretrained`
   adds a small value head on top of the LM for advantage estimation.
   With `--use-lora`, a `LoraConfig` is passed in and PEFT wraps the base.
4. **Reference policy** — under LoRA, *not* loaded as a separate model;
   TRL disables adapters during the reference forward pass. Without LoRA,
   a frozen second copy is loaded.
5. **Tokenizer** — pad token falls back to EOS if the model doesn't ship
   one.
6. **Dataset** — prompt-only. The script tokenizes
   `dataset[args.prompt_field]` to ≤ `--max-prompt-length`. PPO generates
   responses online, so no labels are needed.
7. **PPOConfig dispatch** — `--method clip` sets `cliprange=0.2`,
   `adap_kl_ctrl=False`. `--method kl` sets `cliprange=100` (effectively
   off), `adap_kl_ctrl=True`, `target=--target-kl`.
8. **Reward model** — loaded as an HF `text-classification` pipeline;
   `function_to_apply="none"` so we get raw logits as scalar rewards.

### 4.2 Training loop

Per step:

1. **Sample** a batch of prompts from `ppo_trainer.dataloader`.
2. **Generate** responses with the current policy
   (`ppo_trainer.generate(...)` — uses `max_new_tokens`, `temperature`,
   `top_p`).
3. **Score** each `(prompt + response)` string with the reward model
   pipeline; the scalar logit becomes the reward.
4. **PPO update** via `ppo_trainer.step(query_tensors, response_tensors,
   rewards)`. Internally:
   - Computes log-probs under both the current policy and the reference.
   - Computes advantages from the value head.
   - Runs `--ppo-epochs` mini-batch updates (default 4) over the batch.
   - In Clip mode: clipping bounds the per-token ratio update.
   - In KL mode: an adaptive coefficient adjusts the KL penalty toward
     `--target-kl`.
5. **Log** mean reward + KL every 10 steps; full TRL stats via
   `ppo_trainer.log_stats`.

The loop runs for `--max-steps` (default 200).

### 4.3 Saving

`ppo_trainer.save_pretrained(args.output_dir)` writes either:

```
ppo_finetuned/
├── adapter_config.json        # if LoRA
├── adapter_model.safetensors  # if LoRA
├── tokenizer.json / tokenizer_config.json
└── ...
```

…or a full model checkpoint if LoRA was disabled. Reload with
`AutoModelForCausalLM.from_pretrained(...)` (or `PeftModel.from_pretrained`
on top of the base for adapter-only saves — `inference.py` handles both).

---

## 5. Tuning notes

- **Reward hacking** is the classic PPO failure mode: the policy finds a
  high-reward but degenerate behavior (e.g. always emitting a phrase the
  reward model loves). Watch the KL term — if it grows fast while reward
  also grows fast, the policy is escaping the reference's distribution.
- **`--init-kl-coef`** in Clip mode: too small → policy drifts; too large →
  no learning. 0.05–0.2 is a reasonable sweep.
- **`--target-kl`** in KL mode: smaller = tighter leash. 2–10 is typical.
- **`--ppo-epochs`** (default 4) trades sample efficiency for stability;
  more epochs reuse each batch but risk overfitting to the current rewards.
- **Reward model match** matters more than size. A reward model trained on
  the same domain as your prompts beats a larger generic one.
