"""Fine-tune a Gemma-4 vision model on an image+text dataset using Unsloth + LoRA.

Load a 4-bit Gemma-4 vision model, attach LoRA adapters (vision + language),
convert an image/text dataset into the multimodal chat format, train with
`UnslothVisionDataCollator`, and save the resulting LoRA adapters.
Run with defaults: python gemma4-31b-vision-fine-tune.py. Override e.g. --model-name unsloth/gemma-4-31B-it
  --max-steps -1 --num-train-epochs 2.

Note on model size: the actual model fine-tuned is whatever `--model-name` resolves to.
This script defaults to `unsloth/gemma-4-E4B-it` (not 31B), so it fits
on a single Colab L4. Pass `--model-name unsloth/gemma-4-31B-it` to target
the 31B variant (requires substantially more VRAM).
"""

import argparse
import logging

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel, get_chat_template
from unsloth.trainer import UnslothVisionDataCollator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for paths, model loading, LoRA, and training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a Gemma-4 vision model with Unsloth + LoRA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / output paths
    parser.add_argument(
        "--model-name",
        type=str,
        default="unsloth/gemma-4-E4B-it",
        help="HF repo id or local path of the base vision model to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gemma_4_lora",
        help="Directory where the fine-tuned LoRA adapters and processor will be saved.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs",
        help="Directory used by the trainer for intermediate checkpoints.",
    )

    # Dataset
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="unsloth/LaTeX_OCR",
        help="HF dataset id used for fine-tuning (must contain `image` and `text` fields).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split selector (HF slicing syntax supported).",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Write the LaTeX representation for this image.",
        help="User instruction paired with each training image.",
    )

    # Model loading
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the base model in 4-bit precision.",
    )
    parser.add_argument(
        "--use-gradient-checkpointing",
        type=str,
        default="unsloth",
        help="Gradient checkpointing strategy ('unsloth', 'true', 'false').",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for gated models (optional).",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="gemma-4",
        help="Chat template name passed to unsloth.get_chat_template.",
    )

    # LoRA
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--finetune-vision-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    parser.add_argument("--target-modules", type=str, default="all-linear")
    parser.add_argument(
        "--use-rslora",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    # Training hyperparameters
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Set to -1 to disable and use --num-train-epochs instead.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--optim", type=str, default="adamw_8bit")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-strategy", type=str, default="steps")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        help="Logging integration (e.g. 'none', 'wandb', 'tensorboard').",
    )

    return parser.parse_args()


def load_model_and_processor(args: argparse.Namespace):
    """Load the base vision model in 4-bit, attach LoRA adapters, set the chat template."""
    logger.info("Loading base vision model: %s", args.model_name)
    model, processor = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        token=args.hf_token,
    )

    logger.info("Attaching LoRA adapters (r=%d, alpha=%d).", args.lora_r, args.lora_alpha)
    model = FastVisionModel.get_peft_model(
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
        use_rslora=args.use_rslora,
        loftq_config=None,
        target_modules=args.target_modules,
    )

    processor = get_chat_template(processor, args.chat_template)
    return model, processor


def prepare_dataset(args: argparse.Namespace):
    """Load the dataset and convert each (image, text) pair into a multimodal chat sample."""
    logger.info("Loading dataset: %s [%s]", args.dataset_name, args.dataset_split)
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    # Vision SFT expects a list of {"messages": [...]} samples with image + text turns.
    def convert_to_conversation(sample):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.instruction},
                        {"type": "image", "image": sample["image"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["text"]}],
                },
            ]
        }

    return [convert_to_conversation(sample) for sample in dataset]


def build_trainer(args: argparse.Namespace, model, processor, dataset) -> SFTTrainer:
    """Build an SFTTrainer wired with UnslothVisionDataCollator for multimodal training."""
    sft_config = SFTConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        output_dir=args.checkpoint_dir,
        report_to=args.report_to,
        # Required for vision finetuning: skip text-only dataset prep and let the
        # vision data collator handle image+text batching.
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_length,
    )

    return SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=sft_config,
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

    model, processor = load_model_and_processor(args)
    dataset = prepare_dataset(args)
    trainer = build_trainer(args, model, processor, dataset)

    start_mem = log_memory_stats("Pre-training memory")

    logger.info("Starting training.")
    stats = trainer.train()
    logger.info(
        "Training finished in %.2f minutes.",
        stats.metrics["train_runtime"] / 60,
    )

    log_memory_stats("Post-training memory", baseline_gb=start_mem)

    logger.info("Saving LoRA adapters and processor to: %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
