# LLM Fine-Tuning

A collection of LLM post-training recipes — supervised fine-tuning, continued pre-training, preference optimization, online reinforcement learning, and knowledge distillation — implemented end-to-end and organized so that each technique can be studied (and reused) on its own. Subfolders go from the simplest single-GPU SFT up to multi-GPU DDP/FSDP training and full PPO RLHF loops.

## Introduction: Why several methods, model families, and frameworks?

There is no single "fine-tune the model" step. Different goals, data shapes, and hardware budgets call for different post-training methods, different base models, and different training frameworks. The subfolders here deliberately cover the main combinations so the same project can be looked at from each angle.

### Why several fine-tuning methods (SFT, CPT, DPO/ORPO/KTO, PPO RLHF, GRPO RL, distillation)?

Each method takes a *different kind of supervision signal* and is the right answer to a *different kind of question*:

- **Continued pre-training (CPT / DAPT)** — same next-token loss as the original pre-training run, applied to a domain corpus. Use it to inject *domain priors* (legal, medical, code, biology) before any task-specific tuning. Data shape: plain text. Output: a smarter base model, **not** a chatbot.
- **Supervised fine-tuning (SFT)** — cross-entropy on `(prompt, response)` pairs. The bread-and-butter recipe for teaching a model a specific task or chat format. Variants here cover text, vision, and audio modalities.
- **Preference optimization (DPO / ORPO / KTO)** — train on preference data without an online RL loop. DPO and ORPO use `(prompt, chosen, rejected)` pairs; KTO uses unary thumbs-up/down labels. Cheaper and more stable than PPO, often "good enough."
- **PPO RLHF** — the classic three-model loop: a trainable policy, a frozen reference, a separate reward model. Heaviest setup, but the most expressive — it can optimize *any* reward function the reward model represents.
- **GRPO (group-relative policy optimization)** — an online RL variant that scores generations with *programmatic* reward functions instead of a learned reward model. Useful when correctness is checkable in code (Sudoku solver, unit tests, math verifier).
- **Knowledge distillation** — train a *small* student to mimic a larger teacher's softened output distribution. Unique value: it **shrinks** a model while keeping most of its capability, for serving cost, latency, or on-device deployment.

In short: pick CPT when you have a corpus, SFT when you have task examples, DPO/ORPO/KTO when you have preferences, PPO when you need a learned reward signal, GRPO when correctness is checkable in code, and distillation when you need a smaller model.

### Why several model families (Gemma, Llama, Qwen, etc.)?

- **Gemma-4** (multimodal) — Google's open Gemma family with native text, vision, and audio submodules. Used here for the multimodal SFT examples and the GRPO RL example.
- **Llama 3 / 3.1 / 3.2** — Meta's open-weights flagship. Used as the default for single-GPU SFT, multi-GPU DDP/FSDP, preference optimization, continued pre-training, RLHF, and distillation. Most examples also work with Llama-architecture mirrors and clones (Mistral, Ministral, NousResearch mirrors, Unsloth mirrors).
- **Qwen / Mistral / DeBERTa** — used as alternates or as auxiliary models (DeBERTa as a reward model in the PPO loop, Qwen as a tokenizer-compatible distillation pair).

The point is not the specific model — it is that the same recipe should work for any model in the same architecture family. Examples are written so a one-flag change (`--model-name`, `--teacher-model`) swaps in a different checkpoint.

### Why several frameworks (HF Trainer, TRL, Unsloth, raw PyTorch, vLLM)?

- **Hugging Face `Trainer`** — the default for SFT, CPT, and distillation. Handles mixed precision, gradient accumulation, evaluation, checkpointing, and (with `torchrun`) DDP/FSDP automatically.
- **TRL** — Hugging Face's RL/alignment trainers (`SFTTrainer`, `DPOTrainer`, `ORPOTrainer`, `KTOTrainer`, `PPOTrainer`, `GRPOTrainer`). Used everywhere preference / RL signals are involved.
- **Unsloth** — patched, kernel-fused versions of the training stack. Used for the Gemma-4 multimodal scripts because it gives 2-4x faster training and lower VRAM, especially in 4-bit.
- **PEFT / LoRA** — used across SFT, RLHF, and preference optimization to keep only ~0.2-1% of parameters trainable. Also the trick that makes RLHF tractable on a single GPU: the frozen reference is the same base model with adapters disabled, so there is no second copy in VRAM.
- **Raw PyTorch (from-scratch variants)** — every major recipe (SFT, PPO, DPO, CPT, multi-GPU) ships a `train_from_scratch.py` companion that re-implements the trainer, LR schedule, AMP/scaler, gradient accumulation, GAE, PPO loss, value head, adaptive-KL controller, etc. by hand. Useful for understanding what the `Trainer` does for you.
- **vLLM** — production-grade batched inference with tensor parallelism and paged attention; included in the multi-GPU subfolder for serving the trained models.

### Why several parallelism strategies (DDP, FSDP, `device_map="auto"`, vLLM tensor parallelism)?

- **DDP** — every GPU holds a full copy of the model; gradients are all-reduced. Use when the model fits on a single GPU. Default for the multi-GPU SFT script.
- **FSDP** — model parameters, gradients, and optimizer states are sharded across GPUs. Use when the model does **not** fit on a single GPU (8B+ in fp32, 30B+ in bf16).
- **`device_map="auto"`** (Accelerate) — for *inference only*, automatically shards a too-big model across visible GPUs. The simplest way to run a model that doesn't fit on one card.
- **vLLM tensor parallelism** — the right answer for batched / production inference; several times the throughput of `model.generate`.

## Repository layout

Each subfolder is a self-contained mini-project with its own `README.md`, `requirements.txt`, training script, inference script, and (where relevant) a Docker / serving setup or a from-scratch companion. They are numbered roughly by complexity / progression.

### `1-gemma4-multimodal-supervised-fine-tuning/`
Production-grade Python conversions of the Unsloth Gemma-4 notebooks. Loads a Gemma-4 base model, attaches **LoRA** adapters via Unsloth's `FastModel` / `FastVisionModel`, and runs SFT or GRPO RL via TRL. Four scripts cover the four modalities/algorithms:
- `gemma4-31b-text-supervised-fine-tuning.py` — text SFT on `mlabonne/FineTome-100k`, 4-bit on two GPUs (`device_map="balanced"`), `train_on_responses_only` masking.
- `gemma4-vision-supervised-fine-tuning.py` — vision SFT (LaTeX-OCR) with the `UnslothVisionDataCollator`, LoRA over all-linear modules including vision layers.
- `gemma4-audio-supervised-fine-tuning.py` — audio SFT (German ASR on Emilia-DE), LoRA target modules extended to Gemma-4's audio submodules.
- `gemma4-grpo-reinforcement-learning.py` — GRPO RL with three programmatic reward functions (does the strategy parse, does it avoid third-party imports, does it solve a Sudoku puzzle); base weights frozen, only LoRA adapters updated.

### `2-llama-8b-or-similar-model-supervised-fine-tuning/`
Single-node Llama 3.1 8B SFT with the Hugging Face `Trainer`, plus the deployment story.
- `train.py` — full SFT pipeline: optional layer freezing (`--freeze-base`), optional `clean_text` + `nlpaug` augmentation, optional class-imbalance handling (oversampling + weighted `CrossEntropyLoss`), W&B integration, fp16, 3 epochs by default.
- `train_from_scratch.py` — same CLI, same outputs, but the trainer is re-implemented manually: hand-rolled LR schedule (linear warmup → linear decay), shifted causal-LM cross-entropy (with weighted variant), `torch.cuda.amp` autocast + `GradScaler`, gradient accumulation, gradient clipping, periodic eval (loss + perplexity), checkpoint pruning à la `save_total_limit`, and an `AdamW` built only over parameters that still require gradients.
- `inference.py` — `text-generation` pipeline wrapper.
- `serve.py` + `Dockerfile` — minimal FastAPI `/generate` endpoint; container image based on `python:3.10-slim`.

### `3-multi-gpu-supervised-fine-tuning-and-inference/`
Reference for running training and inference across multiple NVIDIA GPUs.
- `multi-gpu-fine-tuning.py` — full CLI-driven trainer that toggles between **DDP** and **FSDP** via a single flag; spawned with `torchrun --nproc_per_node=N`. Multi-node example included.
- `multi-gpu-fine-tuning-from-scratch.py` — the same idea without `Trainer`: manual `init_process_group`, manual DDP wrap, hand-rolled training loop.
- `multi-gpu-fine-tuning-short.py` — minimal ~50-line scaffold (teaching version).
- `multi-gpu-inference.py` — `device_map="auto"` HF inference; bottom-of-file comments document data / tensor / pipeline parallelism and the DeepSpeed-vs-vLLM decision matrix.
- `multi-gpu-inference-vllm.py` — production-style batched inference with vLLM (tensor parallelism + paged attention).
- `multi-gpu-inference-short.py` — short reference version of the inference script.

### `4-preference-optimization/`
Offline preference-based alignment with TRL — no online rollouts, no reward model, no PPO loop. A single `train.py` dispatches on `--method {dpo,orpo,kto}`:
- **DPO** — contrastive loss on `(prompt, chosen, rejected)` against a frozen reference (the base model with adapters disabled when LoRA is on).
- **ORPO** — SFT loss on the chosen response combined with an odds-ratio penalty on the rejected one. Single-stage; no reference model needed.
- **KTO** — unary `(prompt, completion, label)` data; useful when you only have one-sided thumbs-up/down feedback.
- `train_from_scratch.py` re-implements all three losses by hand; `inference.py` runs the resulting checkpoint.

### `5-continued-pretraining/`
Domain-adaptive pre-training (CPT / DAPT). Same causal-LM next-token loss the base model was trained with, applied to a domain corpus.
- `train.py` — tokenize, group into fixed-length blocks (default 1024), run more pre-training. Default base is `meta-llama/Llama-3.2-1B` (the *base*, non-instruct variant); typical pipeline is `Base → CPT → SFT (→ DPO)`.
- `train_from_scratch.py` — manual block-grouping + manual training loop.
- `inference.py` — generation from the domain-adapted base.

### `6-reinforcement-learning-from-human-feedback-ppo/`
Full RLHF with PPO. Trainable **policy** generates, frozen **reward model** scores, frozen **reference policy** anchors via KL.
- `train.py` — TRL `PPOTrainer`. Two variants behind `--method`:
  - **PPO-Clip** — clips the importance ratio outside `[1-ε, 1+ε]` (the dominant variant in practice).
  - **Adaptive-KL PPO** — clipping disabled, a KL penalty between current and old policy is the main constraint, with the coefficient adapted to keep KL near `--target-kl`.
- `train_from_scratch.py` — same CLI, but the value head, GAE advantage estimation, PPO loss, adaptive-KL controller, and training loop are all implemented manually in pure PyTorch — no TRL.
- `inference.py` — generation from the PPO-tuned policy.
- Default policy is `Llama-3.2-1B-Instruct` with LoRA + DeBERTa reward model so the demo fits on a single mid-range GPU; `--load-in-4bit` / `--load-in-8bit` for larger policies.

### `7-knowledge-distillation/`
Online logit distillation: teacher and student forward-pass on the same batch every step; the student learns to match the teacher's *temperature-softened* token-level distribution (KL) plus the standard cross-entropy on labels.
- `train.py` — a `Trainer` subclass (`DistillationTrainer`) with a custom `compute_loss` that combines `alpha * KL(student || teacher) + (1 - alpha) * CE`, KL computed on shifted positions and scaled by `T²` per Hinton et al.
- The script enforces that the teacher and student share a vocabulary (default pair: Llama-3.1-8B-Instruct → Llama-3.2-1B-Instruct, both on the Llama-3 tokenizer); cross-family distillation requires falling back to offline / response distillation.
- `inference.py` — runs the distilled student, which is a regular HF checkpoint, just smaller.

## Suggested reading order

1. **`2-llama-8b-or-similar-model-supervised-fine-tuning/`** — the simplest end-to-end SFT, plus the from-scratch companion that shows what `Trainer` does for you, and the FastAPI/Docker serving recipe.
2. **`5-continued-pretraining/`** — the simplest *non*-instruction recipe; useful as a baseline for what "fine-tuning" means before chat templates enter the picture.
3. **`1-gemma4-multimodal-supervised-fine-tuning/`** — SFT extended to vision and audio, plus a first taste of online RL (GRPO).
4. **`4-preference-optimization/`** — alignment without rollouts: DPO, ORPO, KTO.
5. **`6-reinforcement-learning-from-human-feedback-ppo/`** — full RLHF with PPO-Clip and adaptive-KL, both via TRL and from scratch.
6. **`7-knowledge-distillation/`** — using a trained model to make a smaller one.
7. **`3-multi-gpu-supervised-fine-tuning-and-inference/`** — the orthogonal axis: how to scale any of the above to multiple GPUs (DDP / FSDP for training, `device_map="auto"` and vLLM for inference).
