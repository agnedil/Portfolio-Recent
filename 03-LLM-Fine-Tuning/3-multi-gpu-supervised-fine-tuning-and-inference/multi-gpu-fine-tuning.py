"""Multi-GPU fine-tuning of a causal LM with HF Trainer + torchrun.

Hardware / strategy notes:
* Multiple NVIDIA GPUs (4x A100, V100, or RTX 4090 class) connected via
  NVLink (NVIDIA's GPU-GPU interconnect) or PCIe (the standard GPU-CPU bus).
* Distributed Data Parallel (DDP) — used when the model fits on 1 GPU.
  Each GPU gets a full replica of the model, the batch is split across
  ranks, and gradients are all-reduced after each backward pass.
  HF `Trainer` enables DDP automatically when launched under `torchrun`.
* Fully Sharded Data Parallel (FSDP) — used when the model is too large
  for a single GPU. Parameters / gradients / optimizer state are sharded
  across all ranks. Toggle with `--fsdp`.
* PyTorch handles synchronization. Launch with:
    torchrun --nproc_per_node=4 multi-gpu-fine-tuning.py
"""

import argparse
import logging

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for paths, dataset, and multi-GPU strategy."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU fine-tuning with HF Trainer + torchrun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / output paths
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--hf-token", type=str, default=None)

    # Dataset
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wikitext",
        help="HF dataset id.",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset config name (e.g. 'wikitext-2-raw-v1' for wikitext).",
    )
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="validation")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--max-length", type=int, default=512)

    # Training hyperparameters
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--evaluation-strategy", type=str, default="steps")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mixed-precision (fp16) training.",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use bf16 instead of fp16 (Ampere+ recommended).",
    )

    # Multi-GPU strategy
    parser.add_argument(
        "--fsdp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use FSDP instead of DDP (set when the model is too large for one GPU).",
    )
    parser.add_argument(
        "--fsdp-transformer-layer",
        type=str,
        default="LlamaDecoderLayer",
        help="Decoder-layer class name to wrap with FSDP.",
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set True only if your model has conditional branches skipping params.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", type=str, default="none")

    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    """Load tokenizer + causal LM; pad with EOS if no pad token is set."""
    logger.info("Loading tokenizer + model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # No `device_map` here — under DDP/FSDP each rank owns its own copy/shard,
    # and Trainer handles placement.
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=args.hf_token)
    return model, tokenizer


def prepare_dataset(args: argparse.Namespace, tokenizer):
    """Load + tokenize the dataset; for causal LM, labels mirror input_ids."""
    logger.info(
        "Loading dataset: %s (config=%s)", args.dataset_name, args.dataset_config
    )
    if args.dataset_config:
        raw = load_dataset(args.dataset_name, args.dataset_config)
    else:
        raw = load_dataset(args.dataset_name)

    def tokenize_fn(examples):
        encodings = tokenizer(
            examples[args.text_field],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        encodings["labels"] = [ids[:] for ids in encodings["input_ids"]]
        return encodings

    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw[args.train_split].column_names,
    )
    eval_ds = tokenized[args.eval_split] if args.eval_split in tokenized else None
    return tokenized[args.train_split], eval_ds


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    """Assemble TrainingArguments; switch to FSDP when --fsdp is set."""
    kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        seed=args.seed,
        report_to=args.report_to,
    )

    if args.fsdp:
        # Shard parameters/gradients/optimizer across ranks at decoder-layer
        # granularity. `auto_wrap` walks the module tree and wraps any module
        # whose class name matches `fsdp_transformer_layer_cls_to_wrap`.
        kwargs["fsdp"] = "full_shard auto_wrap"
        kwargs["fsdp_config"] = {
            "fsdp_transformer_layer_cls_to_wrap": [args.fsdp_transformer_layer],
        }

    return TrainingArguments(**kwargs)


def main() -> None:
    """End-to-end pipeline: load -> tokenize -> train -> save (rank-0 only)."""
    args = parse_args()

    if torch.cuda.is_available():
        logger.info("Visible CUDA devices: %d", torch.cuda.device_count())

    model, tokenizer = load_model_and_tokenizer(args)
    train_ds, eval_ds = prepare_dataset(args, tokenizer)
    training_args = build_training_args(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    logger.info(
        "Starting training (%s strategy).", "FSDP" if args.fsdp else "DDP"
    )
    trainer.train()

    if eval_ds is not None:
        metrics = trainer.evaluate()
        logger.info("Eval metrics: %s", metrics)

    # Trainer.save_model writes from rank 0 only — safe under torchrun.
    logger.info("Saving model + tokenizer to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
