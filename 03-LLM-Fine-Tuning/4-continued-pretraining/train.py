"""Continued pre-training (CPT / DAPT): unsupervised LM on a domain corpus.

Unlike SFT, there's no instruction format, no chat template, no labels file.
The objective is plain causal-LM next-token prediction on raw text — the
same objective the base model was originally trained with — applied to a
domain corpus to make the model "smarter" about that domain.

Pipeline:
1. Load raw text dataset.
2. Tokenize.
3. Group tokens into fixed-length blocks (`--block-size`) so we don't waste
   compute on padding.
4. Train with `Trainer` + `DataCollatorForLanguageModeling(mlm=False)`.
"""

import argparse
import logging
from itertools import chain

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the corpus, model, blocking, and training schedule."""
    parser = argparse.ArgumentParser(
        description="Continued pre-training (CPT) on a domain corpus.",
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
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Length of each training sequence after grouping.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Worker processes for `dataset.map`.",
    )

    # Training hyperparameters
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="CPT typically uses smaller LR than SFT — easy to drift away from base.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--evaluation-strategy", type=str, default="steps")
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", type=str, default="none")
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    """Load tokenizer + model; pad with EOS if no pad token is configured."""
    logger.info("Loading tokenizer + model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=args.hf_token)
    return model, tokenizer


def build_blocked_dataset(args, tokenizer, raw_split):
    """Tokenize text and group into fixed-length blocks (the standard LM-training trick)."""

    def tokenize_fn(examples):
        return tokenizer(examples[args.text_field])

    tokenized = raw_split.map(
        tokenize_fn,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=raw_split.column_names,
    )

    block_size = args.block_size

    # Concatenate all tokens, then split into block_size chunks. This avoids
    # padding waste and matches how base models were originally pre-trained.
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


def main() -> None:
    """End-to-end pipeline: load -> tokenize+block -> train -> save."""
    args = parse_args()

    if torch.cuda.is_available():
        logger.info("Visible CUDA devices: %d", torch.cuda.device_count())

    model, tokenizer = load_model_and_tokenizer(args)

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

    eval_ds = None
    if args.eval_split and args.eval_split in raw:
        eval_ds = build_blocked_dataset(args, tokenizer, raw[args.eval_split])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        bf16=args.bf16,
        seed=args.seed,
        report_to=args.report_to,
    )

    # mlm=False -> standard left-to-right causal LM (next-token prediction).
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    logger.info("Starting CPT.")
    trainer.train()

    if eval_ds is not None:
        metrics = trainer.evaluate()
        logger.info("Eval metrics: %s", metrics)

    logger.info("Saving model + tokenizer to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
