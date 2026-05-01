"""Continued pre-training (CPT / DAPT) — pure PyTorch, no HF Trainer.

Same workflow and CLI as `train.py`, but the training infrastructure is
hand-rolled:

* `build_blocked_dataset`     — tokenize and group into fixed-length blocks
                                (the canonical LM-training packing trick)
* `make_dataloader`           — DataLoader with a tiny tensor-stacking collate
* `causal_lm_loss`            — shifted cross-entropy
* `cosine_schedule_lr`        — manual warmup + cosine (or linear) decay
* `train_loop`                — bf16 autocast + grad accumulation + LR
                                schedule + periodic eval + checkpoints
* `evaluate`                  — eval loss + perplexity
* `save_checkpoint` / `prune` — match `save_total_limit` behavior

HF Transformers + datasets are still used for model / tokenizer / corpus
loading (those aren't the algorithm). Optional bf16, gradient accumulation,
cosine schedule, evaluation strategy, and save-limit pruning are all
preserved from the Trainer-based version.
"""

import argparse
import logging
import math
import shutil
from itertools import chain
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# CLI (matches train.py + a few from-scratch-only knobs)
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Same flags as `train.py` so the two scripts are interchangeable."""
    parser = argparse.ArgumentParser(
        description="Continued pre-training in pure PyTorch (no HF Trainer).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / output paths
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-dir", type=str, default="./cpt_finetuned")
    parser.add_argument("--hf-token", type=str, default=None)

    # Corpus
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="validation")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--max-train-samples", type=int, default=None)

    # Tokenization / blocking
    parser.add_argument("--block-size", type=int, default=1024,
                        help="Length of each training sequence after grouping.")
    parser.add_argument("--num-proc", type=int, default=4,
                        help="Worker processes for dataset.map.")

    # Training hyperparameters
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="CPT typically uses smaller LR than SFT — easy to drift away from base.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine",
                        choices=["cosine", "linear"])
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--evaluation-strategy", type=str, default="steps",
                        choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ------------------------------------------------------------------
# Data: tokenize + group into fixed-length blocks
# ------------------------------------------------------------------

def build_blocked_dataset(args, tokenizer, raw_split):
    """Tokenize raw text and pack into `--block-size` chunks (no padding waste)."""

    def tokenize_fn(examples):
        return tokenizer(examples[args.text_field])

    tokenized = raw_split.map(
        tokenize_fn,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=raw_split.column_names,
    )

    block_size = args.block_size

    # Concatenate all tokens, drop the trailing remainder, slice into blocks.
    # Matches the way base models were originally pre-trained.
    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = (len(concatenated[list(examples.keys())[0]]) // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = [ids[:] for ids in result["input_ids"]]
        return result

    return tokenized.map(group_texts, batched=True, num_proc=args.num_proc)


def make_dataloader(dataset, batch_size, shuffle, num_workers):
    """DataLoader over the HF Dataset with a collate that stacks tensors."""

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
        pin_memory=True,
    )


# ------------------------------------------------------------------
# Loss + LR schedule
# ------------------------------------------------------------------

def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shifted cross-entropy — matches HF's internal causal LM loss."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def schedule_lr(step: int, warmup_steps: int, total_steps: int, kind: str) -> float:
    """Manual warmup → cosine (default) or linear decay multiplier in [0, 1]."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    if kind == "linear":
        return max(0.0, 1.0 - progress)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ------------------------------------------------------------------
# Eval + checkpointing
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, eval_loader, device, bf16: bool):
    """Run an eval pass; return mean loss + perplexity."""
    if eval_loader is None:
        return {}
    model.eval()
    autocast_kwargs = (
        dict(enabled=torch.cuda.is_available(), dtype=torch.bfloat16)
        if bf16 else dict(enabled=False)
    )
    total_loss, total_batches = 0.0, 0
    for batch in eval_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.cuda.amp.autocast(**autocast_kwargs):
            out = model(input_ids=batch["input_ids"])
            loss = causal_lm_loss(out.logits, batch["labels"])
        total_loss += loss.item()
        total_batches += 1
    model.train()
    avg = total_loss / max(1, total_batches)
    return {"eval_loss": avg, "eval_ppl": math.exp(avg) if avg < 30 else float("inf")}


def save_checkpoint(model, tokenizer, output_dir: str, step: int):
    """Write `output_dir/checkpoint-{step}/` and return the path."""
    ckpt = Path(output_dir) / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    logger.info("Saved checkpoint to %s", ckpt)
    return str(ckpt)


def prune_old_checkpoints(output_dir: str, save_total_limit: int):
    """Match HF Trainer's save_total_limit by deleting oldest checkpoints first."""
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
# Training loop
# ------------------------------------------------------------------

def train_loop(model, tokenizer, train_loader, eval_loader, optimizer, args, device):
    """Manual loop: bf16 autocast + grad accumulation + LR schedule + eval + ckpt."""
    autocast_kwargs = (
        dict(enabled=torch.cuda.is_available(), dtype=torch.bfloat16)
        if args.bf16 else dict(enabled=False)
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = max(1, int(steps_per_epoch * args.num_train_epochs))
    warmup_steps = int(total_steps * args.warmup_ratio)

    logger.info(
        "Training: %d optimizer steps over %.2f epochs (grad_accum=%d, warmup=%d, schedule=%s).",
        total_steps, args.num_train_epochs, args.gradient_accumulation_steps,
        warmup_steps, args.lr_scheduler_type,
    )

    model.train()
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    epochs_to_run = max(1, math.ceil(args.num_train_epochs))
    for epoch in range(epochs_to_run):
        for batch in train_loader:
            if global_step >= total_steps:
                break

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.cuda.amp.autocast(**autocast_kwargs):
                out = model(input_ids=batch["input_ids"])
                loss = causal_lm_loss(out.logits, batch["labels"])
                # Scale by 1/accum so the effective grad is the average across micro-batches.
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            running_loss += loss.item()
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps != 0:
                continue

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )

            lr_scale = schedule_lr(global_step, warmup_steps, total_steps, args.lr_scheduler_type)
            for pg in optimizer.param_groups:
                pg["lr"] = args.learning_rate * lr_scale

            optimizer.step()
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

            # ---- Eval ----
            if (
                eval_loader is not None
                and args.evaluation_strategy == "steps"
                and global_step % args.eval_steps == 0
            ):
                metrics = evaluate(model, eval_loader, device, args.bf16)
                logger.info("eval @ step=%d: %s", global_step, metrics)

            # ---- Checkpoint ----
            if global_step % args.save_steps == 0:
                save_checkpoint(model, tokenizer, args.output_dir, global_step)
                prune_old_checkpoints(args.output_dir, args.save_total_limit)

        if global_step >= total_steps:
            break

        if eval_loader is not None and args.evaluation_strategy == "epoch":
            metrics = evaluate(model, eval_loader, device, args.bf16)
            logger.info("eval @ epoch=%d: %s", epoch, metrics)

    return global_step


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        logger.info("Visible CUDA devices: %d", torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading tokenizer + model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=args.hf_token,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    model.to(device)

    logger.info("Loading dataset: %s/%s", args.dataset_name, args.dataset_config)
    raw = (
        load_dataset(args.dataset_name, args.dataset_config)
        if args.dataset_config
        else load_dataset(args.dataset_name)
    )

    train_raw = raw[args.train_split]
    if args.max_train_samples:
        train_raw = train_raw.select(range(min(args.max_train_samples, len(train_raw))))
    train_ds = build_blocked_dataset(args, tokenizer, train_raw)
    train_loader = make_dataloader(
        train_ds, args.per_device_train_batch_size, shuffle=True, num_workers=args.num_workers,
    )

    eval_loader = None
    if args.eval_split and args.eval_split in raw:
        eval_ds = build_blocked_dataset(args, tokenizer, raw[args.eval_split])
        eval_loader = make_dataloader(
            eval_ds, args.per_device_eval_batch_size, shuffle=False, num_workers=args.num_workers,
        )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting CPT.")
    final_step = train_loop(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        args=args,
        device=device,
    )

    if eval_loader is not None and args.evaluation_strategy != "no":
        metrics = evaluate(model, eval_loader, device, args.bf16)
        logger.info("Final eval: %s", metrics)

    logger.info("Saving final model + tokenizer to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done. Trained for %d optimizer steps.", final_step)


if __name__ == "__main__":
    main()
