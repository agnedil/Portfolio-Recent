"""Fine-tune a Gemma-4 model on a conversational dataset using Unsloth + LoRA.

Load a 4-bit Gemma-4 base model, attach LoRA adapters, format the dataset
with the Gemma-4 chat template, train on assistant responses only, and save
the resulting LoRA adapters.
Run with defaults: python gemma4-31b-text-fine-tune.py. Override anything, e.g. --model-name
  unsloth/gemma-4-E4B-it --max-steps -1 --num-train-epochs 1

Note on model size: the actual model size fine-tuned is whatever `--model-name` resolves to.
This script defaults to `unsloth/gemma-4-31B-it` — genuinely the 31B variant.
Override `--model-name` to target a different size.
"""

import argparse
import logging

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    train_on_responses_only,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for paths, model loading, LoRA, and training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a Gemma-4 model with Unsloth + LoRA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / output paths
    parser.add_argument(
        "--model-name",
        type=str,
        default="unsloth/gemma-4-31B-it",
        help="HF repo id or local path of the base model to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gemma_4_lora",
        help="Directory where the fine-tuned LoRA adapters and tokenizer will be saved.",
    )

    # Dataset
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mlabonne/FineTome-100k",
        help="HF dataset id used for fine-tuning.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train[:3000]",
        help="Dataset split selector (HF slicing syntax supported).",
    )

    # Model loading
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the base model in 4-bit precision.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="balanced",
        help="HF device_map strategy (e.g. 'auto', 'balanced').",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for gated models (optional).",
    )

    # LoRA
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--finetune-vision-layers",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--finetune-language-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--finetune-attention-modules",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--finetune-mlp-modules",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Training hyperparameters
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Set to -1 to disable and use --num-train-epochs instead.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear")
    parser.add_argument("--optim", type=str, default="adamw_8bit")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        help="Logging integration (e.g. 'none', 'wandb', 'tensorboard').",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="gemma-4-thinking",
        help="Chat template name passed to unsloth.get_chat_template.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    """Load the base model in 4-bit, attach LoRA adapters, set the chat template."""
    logger.info("Loading base model: %s", args.model_name)
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        dtype=None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
        token=args.hf_token,
        device_map=args.device_map,
    )

    logger.info("Attaching LoRA adapters (r=%d, alpha=%d).", args.lora_r, args.lora_alpha)
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)
    return model, tokenizer


def prepare_dataset(args: argparse.Namespace, tokenizer):
    """Load the dataset and render each conversation into a `text` field via the chat template."""
    logger.info("Loading dataset: %s [%s]", args.dataset_name, args.dataset_split)
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset = standardize_data_formats(dataset)

    # The Gemma-4 processor adds a single <bos> at training time, so strip any
    # template-injected <bos> to avoid duplicates.
    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            ).removeprefix("<bos>")
            for convo in examples["conversations"]
        ]
        return {"text": texts}
    return dataset.map(formatting_prompts_func, batched=True)


def build_trainer(args: argparse.Namespace, model, tokenizer, dataset) -> SFTTrainer:
    """Build an SFTTrainer that computes loss only on assistant responses."""
    sft_config = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        report_to=args.report_to,
        output_dir=args.output_dir,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=sft_config,
    )

    # Mask the user prompt so loss is only computed on assistant responses.
    return train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )


def log_memory_stats(prefix: str, baseline_gb: float | None = None) -> float:
    """Log peak reserved GPU memory; return it so callers can diff before/after training."""
    if not torch.cuda.is_available():
        return 0.0
    used_gb = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    total_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3)
    logger.info("%s: peak reserved %.3f / %.3f GB", prefix, used_gb, total_gb)
    if baseline_gb is not None:
        logger.info("%s: peak attributable to training %.3f GB", prefix, used_gb - baseline_gb)
    return used_gb


def main() -> None:
    """End-to-end pipeline: load -> prepare data -> train -> save adapters."""
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(args)
    dataset = prepare_dataset(args, tokenizer)
    trainer = build_trainer(args, model, tokenizer, dataset)

    start_mem = log_memory_stats("Pre-training memory")

    logger.info("Starting training.")
    stats = trainer.train()
    logger.info(
        "Training finished in %.2f minutes.",
        stats.metrics["train_runtime"] / 60,
    )

    log_memory_stats("Post-training memory", baseline_gb=start_mem)

    logger.info("Saving LoRA adapters and tokenizer to: %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
