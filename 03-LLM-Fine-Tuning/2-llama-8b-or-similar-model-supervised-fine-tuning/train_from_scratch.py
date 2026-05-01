"""Fine-tune a Llama 3.1 8B model — same workflow as `train.py`, no HF Trainer.

Mirrors the CLI of `train.py` so the two scripts are interchangeable, but
every training-loop component is implemented manually in pure PyTorch:

* `linear_schedule_lr`        — linear warmup + linear decay
* `causal_lm_loss`            — shifted-CE loss (matches HF's internal one)
* `weighted_causal_lm_loss`   — same with a per-class weight (article snippet)
* `make_dataloader`           — DataLoader with a collate that stacks tensors
* `train_loop`                — gradient accumulation, AMP scaler, scheduler,
                                periodic logging, eval, checkpointing
* `evaluate`                  — average loss + perplexity on the eval split
* `save_checkpoint` / `prune` — match the `save_total_limit` behavior

HF Transformers + datasets are still used for model / tokenizer / dataset
loading (those aren't the algorithm). Optional layer freezing, mixed
precision, gradient accumulation, oversampling, augmentation, and wandb
are all preserved from `train.py`.
"""

import argparse
import json
import logging
import math
import os
import re
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# CLI (matches train.py + a few from-scratch-only knobs)
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments — mirrors `train.py` so they're interchangeable."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3.1 8B in pure PyTorch (no HF Trainer).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / output paths
    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", type=str, default="./llama_finetuned")
    parser.add_argument("--logging-dir", type=str, default="./logs")
    parser.add_argument("--hf-token", type=str, default=None)

    # Dataset
    parser.add_argument("--dataset-name", type=str, default="squad")
    parser.add_argument("--jsonl-path", type=str, default=None,
                        help="Local JSONL with `input`/`output` fields. Overrides --dataset-name.")
    parser.add_argument("--text-field", type=str, default="context")
    parser.add_argument("--eval-split-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--clean-text", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--oversample-label", type=str, default=None)
    parser.add_argument("--oversample-target-size", type=int, default=0)

    # Model setup
    parser.add_argument("--freeze-base", action=argparse.BooleanOptionalAction, default=False)

    # Training hyperparameters (article values as defaults)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Linear-warmup steps before linear decay kicks in.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--evaluation-strategy", type=str, default="steps",
                        choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-steps", type=int, default=None,
                        help="Defaults to --save-steps when unset.")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weights", type=float, nargs="+", default=None)

    # Integrations
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    return parser.parse_args()


# ------------------------------------------------------------------
# Data prep helpers (preserved from train.py)
# ------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase and strip non-alphanumeric characters."""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower().strip()


def oversample(dataset, label_value: str, target_size: int):
    """Oversample rows with `label == label_value` up to `target_size` extra rows."""
    from sklearn.utils import resample

    minority = [s for s in dataset if s.get("label") == label_value]
    if not minority:
        logger.warning("No rows match label=%s; skipping oversampling.", label_value)
        return dataset
    extra = resample(minority, replace=True, n_samples=target_size)
    combined = list(dataset) + list(extra)
    return Dataset.from_list(combined)


def augment_texts(texts):
    """Apply nlpaug SynonymAug to each text."""
    import nlpaug.augmenter.word as naw

    augmenter = naw.SynonymAug()
    return [augmenter.augment(t) for t in texts]


def load_and_prepare_dataset(args: argparse.Namespace, tokenizer):
    """Load + (optionally clean / oversample / augment) + tokenize the dataset."""
    if args.jsonl_path:
        logger.info("Loading JSONL dataset from %s", args.jsonl_path)
        raw = load_dataset("json", data_files=args.jsonl_path, split="train")
        text_field = "input"
    else:
        logger.info("Loading HF dataset: %s", args.dataset_name)
        raw = load_dataset(args.dataset_name, split="train")
        text_field = args.text_field

    if args.oversample_label and args.oversample_target_size > 0:
        raw = oversample(raw, args.oversample_label, args.oversample_target_size)

    texts = list(raw[text_field])
    if args.clean_text:
        texts = [clean_text(t) for t in texts]
    if args.augment:
        texts = augment_texts(texts)

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
    )
    encodings["labels"] = [ids[:] for ids in encodings["input_ids"]]
    tokenized = Dataset.from_dict(encodings)
    split = tokenized.train_test_split(test_size=args.eval_split_ratio, seed=args.seed)
    return split["train"], split["test"]


def make_dataloader(dataset, batch_size, shuffle, num_workers):
    """Wrap an HF Dataset with a collate function that stacks tensors."""

    def collate(batch):
        return {
            key: torch.tensor([sample[key] for sample in batch], dtype=torch.long)
            for key in batch[0]
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
    )


# ------------------------------------------------------------------
# Loss + LR scheduler (from scratch)
# ------------------------------------------------------------------

def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard shifted cross-entropy for causal LM (matches HF's internal loss)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Same as `causal_lm_loss`, but with per-class weights (article snippet)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        weight=class_weights.to(shift_logits.device),
        ignore_index=-100,
    )


def linear_schedule_lr(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup to 1.0, then linear decay to 0.0 — matches HF's default."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


# ------------------------------------------------------------------
# Eval + checkpoint helpers (from scratch)
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, eval_loader, device, fp16: bool, class_weights):
    """Run a single eval pass; return mean loss and perplexity."""
    model.eval()
    total_loss, total_batches = 0.0, 0
    autocast = torch.cuda.amp.autocast(dtype=torch.float16, enabled=fp16 and torch.cuda.is_available())
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            if class_weights is not None:
                loss = weighted_causal_lm_loss(out.logits, batch["labels"], class_weights)
            else:
                loss = causal_lm_loss(out.logits, batch["labels"])
        total_loss += loss.item()
        total_batches += 1
    model.train()
    avg = total_loss / max(1, total_batches)
    return {"eval_loss": avg, "eval_ppl": math.exp(avg) if avg < 30 else float("inf")}


def save_checkpoint(model, tokenizer, output_dir: str, step: int):
    """Save model + tokenizer under `output_dir/checkpoint-{step}/`."""
    ckpt = Path(output_dir) / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    logger.info("Saved checkpoint to %s", ckpt)
    return str(ckpt)


def prune_old_checkpoints(output_dir: str, save_total_limit: int):
    """Match HF Trainer's `save_total_limit`: keep only the most recent N."""
    if save_total_limit <= 0:
        return
    ckpts = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    while len(ckpts) > save_total_limit:
        oldest = ckpts.pop(0)
        shutil.rmtree(oldest, ignore_errors=True)


# ------------------------------------------------------------------
# Training loop (from scratch)
# ------------------------------------------------------------------

def train_loop(
    model,
    tokenizer,
    train_loader,
    eval_loader,
    optimizer,
    args,
    device,
    class_weights,
    wandb_run,
):
    """Manual training loop: AMP + grad accumulation + LR schedule + eval + checkpoints."""
    use_amp = args.fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = max(1, int(steps_per_epoch * args.num_train_epochs))
    eval_steps = args.eval_steps or args.save_steps

    logger.info(
        "Training: %d optimizer steps over %.2f epochs (grad_accum=%d, steps/epoch=%d).",
        total_steps, args.num_train_epochs, args.gradient_accumulation_steps, steps_per_epoch,
    )

    model.train()
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(math.ceil(args.num_train_epochs)):
        for batch in train_loader:
            if global_step >= total_steps:
                break

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # ---- Forward + loss (AMP context for fp16) ----
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                if class_weights is not None:
                    loss = weighted_causal_lm_loss(out.logits, batch["labels"], class_weights)
                else:
                    loss = causal_lm_loss(out.logits, batch["labels"])
                # Scale by 1/accum so the effective grad is the average across micro-batches.
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            running_loss += loss.item()
            micro_step += 1

            # Only step the optimizer every `gradient_accumulation_steps` micro-batches.
            if micro_step % args.gradient_accumulation_steps != 0:
                continue

            # ---- Gradient clipping (unscale first, then clip) ----
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )

            # ---- LR schedule applied via the lambda before .step() ----
            lr_scale = linear_schedule_lr(global_step, args.warmup_steps, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = args.learning_rate * lr_scale

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # ---- Logging ----
            if global_step % args.logging_steps == 0:
                avg = running_loss / args.logging_steps
                running_loss = 0.0
                logger.info(
                    "epoch=%d step=%d loss=%.4f lr=%.2e",
                    epoch, global_step, avg, args.learning_rate * lr_scale,
                )
                if wandb_run is not None:
                    wandb_run.log({"train_loss": avg, "lr": args.learning_rate * lr_scale}, step=global_step)

            # ---- Eval ----
            run_eval = (
                eval_loader is not None
                and args.evaluation_strategy == "steps"
                and global_step % eval_steps == 0
            )
            if run_eval:
                metrics = evaluate(model, eval_loader, device, args.fp16, class_weights)
                logger.info("eval @ step=%d: %s", global_step, metrics)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=global_step)

            # ---- Checkpoint ----
            if global_step % args.save_steps == 0:
                save_checkpoint(model, tokenizer, args.output_dir, global_step)
                prune_old_checkpoints(args.output_dir, args.save_total_limit)

        if global_step >= total_steps:
            break

        # End-of-epoch evaluation (when strategy="epoch").
        if eval_loader is not None and args.evaluation_strategy == "epoch":
            metrics = evaluate(model, eval_loader, device, args.fp16, class_weights)
            logger.info("eval @ epoch=%d: %s", epoch, metrics)
            if wandb_run is not None:
                wandb_run.log({**metrics, "epoch": epoch}, step=global_step)

    return global_step


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def validate_gpu() -> None:
    """Log CUDA availability + device name."""
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("Device: %s", torch.cuda.get_device_name(0))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    validate_gpu()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading tokenizer + model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(args.model_name, token=args.hf_token)
    model.to(device)

    if args.freeze_base:
        logger.info("Freezing base model parameters (only the LM head will train).")
        for param in model.base_model.parameters():
            param.requires_grad = False

    train_ds, eval_ds = load_and_prepare_dataset(args, tokenizer)
    train_loader = make_dataloader(
        train_ds, args.per_device_train_batch_size, shuffle=True, num_workers=args.num_workers,
    )
    eval_loader = make_dataloader(
        eval_ds, args.per_device_train_batch_size, shuffle=False, num_workers=args.num_workers,
    )

    # AdamW only on parameters that still require gradients (so --freeze-base
    # doesn't allocate optimizer states for frozen weights).
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    class_weights = (
        torch.tensor(args.class_weights, dtype=torch.float)
        if args.class_weights else None
    )

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.logging_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting training.")
    final_step = train_loop(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        args=args,
        device=device,
        class_weights=class_weights,
        wandb_run=wandb_run,
    )

    if eval_loader is not None and args.evaluation_strategy != "no":
        final_metrics = evaluate(model, eval_loader, device, args.fp16, class_weights)
        logger.info("Final eval: %s", final_metrics)

    logger.info("Saving final model to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done. Trained for %d optimizer steps.", final_step)


if __name__ == "__main__":
    main()
