"""Fine-tune a Llama 3.1 8B model with the Hugging Face Trainer.

Implements the following workflow:
GPU validation, dataset loading, optional cleaning / oversampling /
augmentation, tokenization, optional layer freezing, mixed-precision training,
gradient accumulation, optional Weights & Biases logging, and saving.

Note on model size: defaults to `meta-llama/Meta-Llama-3.1-8B-Instruct`. Full
fine-tuning of an 8B model needs significant VRAM (~80GB+); on smaller GPUs
use `--freeze-base` (the article's approach for limited memory) or swap in a
smaller variant via `--model-name`.

This script is model-agnostic (uses LlamaForCausalLM / AutoTokenizer.from_pretrained), so you only need to
override --model-name. The newest 8B in the Llama line is Llama 3.1 8B (July 2024, 128K context, multilingual).
Llama 3.2 dropped the 8B size (only 1B/3B text + 11B/90B vision shipped), and Llama 4 is MoE-only — so 3.1 is the
most recent dense 8B.

python train.py --model-name meta-llama/Llama-3.1-8B-Instruct                                                     
# or the base (non-instruct) variant:
python train.py --model-name meta-llama/Llama-3.1-8B
                                                           
Things to bump alongside it:                
                                                                                                                   
- --max-length — the tokenizer supports 128K, so raise from the article's 512 if your data and VRAM allow.
- Check tokenizer.pad_token — Llama 3.1 still ships without one; the script already falls back to EOS, so no      
action needed.
- Access — both 3.0 and 3.1 are gated; huggingface-cli login (or --hf-token) must have been granted access to the
specific repo.

Other 8B alternatives the same script can fine-tune without code changes: NousResearch/Meta-Llama-3.1-8B-Instruct
(un-gated mirror), unsloth/llama-3.1-8b-Instruct (faster loads), or any third-party Llama-architecture 8B like    
mistralai/Ministral-8B-Instruct-2410. 
"""

import argparse
import json
import logging
import re
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for paths, dataset prep, training, and integrations."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3.1 8B with the Hugging Face Trainer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / output paths
    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", type=str, default="./llama_finetuned")
    parser.add_argument("--logging-dir", type=str, default="./logs")
    parser.add_argument("--hf-token", type=str, default=None)

    # Dataset
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="squad",
        help="HF dataset id (used when --jsonl-path is not set).",
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default=None,
        help="Optional local JSONL file with `input`/`output` fields. Overrides --dataset-name.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="context",
        help="Field used as training text when loading an HF dataset.",
    )
    parser.add_argument("--eval-split-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--clean-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply regex cleaning (lowercasing, strip non-alphanumerics) before tokenization.",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply nlpaug SynonymAug to training texts (requires `nlpaug`).",
    )
    parser.add_argument(
        "--oversample-label",
        type=str,
        default=None,
        help="If set, oversample rows whose `label` field equals this value.",
    )
    parser.add_argument("--oversample-target-size", type=int, default=0)

    # Model setup
    parser.add_argument(
        "--freeze-base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze base model parameters (only train the LM head).",
    )

    # Training hyperparameters (article values as defaults)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--evaluation-strategy", type=str, default="steps")
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable mixed-precision (fp16) training.",
    )
    parser.add_argument(
        "--class-weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-class weights for a weighted CrossEntropyLoss head.",
    )

    # Integrations
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    return parser.parse_args()


# ---------- Data prep helpers (from the article) ----------

def clean_text(text: str) -> str:
    """Lowercase and strip non-alphanumeric characters."""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower().strip()


def preprocess_jsonl(input_file: str, output_file: str, response_fn) -> None:
    """Read raw lines, clean, attach generated responses, and write JSONL pairs."""
    processed = []
    with open(input_file, "r") as f:
        for line in f:
            input_text = clean_text(line.strip())
            output_text = response_fn(input_text)
            processed.append({"input": input_text, "output": output_text})
    with open(output_file, "w") as f:
        for entry in processed:
            f.write(json.dumps(entry) + "\n")


def oversample(dataset, label_value: str, target_size: int):
    """Oversample rows with `label == label_value` up to `target_size` extra rows."""
    from sklearn.utils import resample  # imported lazily — only needed when used

    minority = [s for s in dataset if s.get("label") == label_value]
    if not minority:
        logger.warning("No rows match label=%s; skipping oversampling.", label_value)
        return dataset
    extra = resample(minority, replace=True, n_samples=target_size)
    combined = list(dataset) + list(extra)
    return Dataset.from_list(combined)


def augment_texts(texts: list[str]) -> list[str]:
    """Apply nlpaug SynonymAug to each text."""
    import nlpaug.augmenter.word as naw  # imported lazily — heavy dependency

    augmenter = naw.SynonymAug()
    return [augmenter.augment(t) for t in texts]


# ---------- Pipeline ----------

def validate_gpu() -> None:
    """Log CUDA availability + device name (article's GPU validation step)."""
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("Device: %s", torch.cuda.get_device_name(0))


def load_and_prepare_dataset(args: argparse.Namespace, tokenizer):
    """Load dataset (JSONL or HF), apply optional cleaning/oversampling/augmentation, tokenize."""
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

    # Tokenize a single text column; labels = input_ids for causal LM.
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
    )
    encodings["labels"] = [ids[:] for ids in encodings["input_ids"]]

    tokenized = Dataset.from_dict(encodings)
    split = tokenized.train_test_split(test_size=args.eval_split_ratio, seed=42)
    return split["train"], split["test"]


class WeightedLossTrainer(Trainer):
    """Trainer subclass that applies a fixed-weight CrossEntropyLoss (article snippet)."""

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    """End-to-end pipeline: validate GPU -> load -> tokenize -> train -> save."""
    args = parse_args()
    validate_gpu()

    logger.info("Loading tokenizer + model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(args.model_name, token=args.hf_token)

    if args.freeze_base:
        logger.info("Freezing base model parameters (only the LM head will train).")
        for param in model.base_model.parameters():
            param.requires_grad = False

    train_ds, eval_ds = load_and_prepare_dataset(args, tokenizer)

    report_to = []
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        report_to = ["wandb"]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_dir=args.logging_dir,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        report_to=report_to,
    )

    if args.class_weights:
        weights = torch.tensor(args.class_weights, dtype=torch.float)
        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            class_weights=weights,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
        )

    logger.info("Starting training.")
    trainer.train()

    logger.info("Saving fine-tuned model to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
