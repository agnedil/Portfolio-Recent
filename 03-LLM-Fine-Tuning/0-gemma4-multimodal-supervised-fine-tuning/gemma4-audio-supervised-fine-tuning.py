"""Fine-tune a Gemma-4 model on an audio+text (ASR) dataset using Unsloth + LoRA.

Load a 4-bit Gemma-4 multimodal model, attach LoRA adapters that include the
audio submodules, format an ASR dataset into the multimodal chat format,
train with `UnslothVisionDataCollator` (which also handles audio), and save
the resulting LoRA adapters.
Run with defaults: python gemma4-31b-audio-fine-tune.py. Override e.g.
--model-name unsloth/gemma-4-31B-it --max-steps -1 --num-train-epochs 1.

Note on model size: the actual model size fine-tuned is whatever `--model-name` resolves to.
This script defaults to `unsloth/gemma-4-E4B-it` (not 31B), so it fits
on a single Colab T4. Pass `--model-name unsloth/gemma-4-31B-it` to target
the 31B variant (requires substantially more VRAM).
"""

import argparse
import logging

import torch
from datasets import Audio, load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel
from unsloth.trainer import UnslothVisionDataCollator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Default LoRA target modules: language layers + Gemma-4 audio submodules.
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    # Audio layers
    "post", "linear_start", "linear_end",
    "embedding_projection",
    "ffw_layer_1", "ffw_layer_2",
    "output_proj",
]

DEFAULT_SYSTEM_PROMPT = "You are an assistant that transcribes speech accurately."
DEFAULT_USER_INSTRUCTION = "Please transcribe this audio."


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for paths, model loading, LoRA, and training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a Gemma-4 audio model with Unsloth + LoRA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / output paths
    parser.add_argument(
        "--model-name",
        type=str,
        default="unsloth/gemma-4-E4B-it",
        help="HF repo id or local path of the base model to fine-tune.",
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
        default="kadirnar/Emilia-DE-B000000",
        help="HF dataset id used for fine-tuning (must contain `audio` and `text` fields).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split selector (HF slicing syntax supported).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3000,
        help="Number of samples to keep from the split (-1 = use all).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Target audio sampling rate for the cast `audio` column.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
    )
    parser.add_argument(
        "--user-instruction",
        type=str,
        default=DEFAULT_USER_INSTRUCTION,
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
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for gated models (optional).",
    )

    # LoRA
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
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
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=DEFAULT_TARGET_MODULES,
        help="LoRA target module names (language + audio submodules by default).",
    )
    parser.add_argument(
        "--use-rslora",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    # Training hyperparameters
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Set to -1 to disable and use --num-train-epochs instead.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--optim", type=str, default="adamw_8bit")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-strategy", type=str, default="steps")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--map-num-proc", type=int, default=4)
    parser.add_argument("--map-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        help="Logging integration (e.g. 'none', 'wandb', 'tensorboard').",
    )
    return parser.parse_args()


def load_model_and_processor(args: argparse.Namespace):
    """Load the base model in 4-bit and attach LoRA adapters covering audio submodules."""
    logger.info("Loading base model: %s", args.model_name)
    model, processor = FastModel.from_pretrained(
        model_name=args.model_name,
        dtype=None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
        token=args.hf_token,
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
        use_rslora=args.use_rslora,
        loftq_config=None,
        target_modules=args.target_modules,
    )
    return model, processor


def prepare_dataset(args: argparse.Namespace):
    """Load the ASR dataset, resample audio, and convert each sample into a multimodal chat."""
    logger.info("Loading dataset: %s [%s]", args.dataset_name, args.dataset_split)
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    if args.num_samples > 0:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    dataset = dataset.cast_column("audio", Audio(sampling_rate=args.sampling_rate))

    # Audio SFT expects a list of {"messages": [...]} samples with audio + text turns.
    def format_audio_data(samples: dict) -> dict:
        formatted = {"messages": []}
        for idx in range(len(samples["audio"])):
            audio_array = samples["audio"][idx]["array"]
            label = str(samples["text"][idx])
            formatted["messages"].append([
                {
                    "role": "system",
                    "content": [{"type": "text", "text": args.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_array},
                        {"type": "text", "text": args.user_instruction},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": label}],
                },
            ])
        return formatted

    return dataset.map(
        format_audio_data,
        batched=True,
        batch_size=args.map_batch_size,
        num_proc=args.map_num_proc,
    )


def build_trainer(args: argparse.Namespace, model, processor, dataset) -> SFTTrainer:
    """Build an SFTTrainer wired with UnslothVisionDataCollator for multimodal (audio) training."""
    sft_config = SFTConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
        # Required for audio finetuning: skip text-only dataset prep and let the
        # vision/audio data collator handle multimodal batching.
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
