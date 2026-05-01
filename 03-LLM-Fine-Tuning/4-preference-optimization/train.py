"""Preference-optimization fine-tuning: DPO / ORPO / KTO via TRL.

All three are alignment methods that train on human (or AI) preferences,
without the online rollouts of RL. They share the same goal — make the model
prefer "good" over "bad" outputs — but differ in mechanics:

* DPO  (Direct Preference Optimization): contrastive loss on
        (prompt, chosen, rejected) pairs. Needs a frozen reference model
        (or runs reference-free via PEFT).
* ORPO (Odds-Ratio Preference Optimization): same (chosen, rejected)
        pairs as DPO, but folds SFT and preference learning into one stage
        and is *reference-free* (no ref_model needed).
* KTO  (Kahneman-Tversky Optimization): unary labels — each row is
        (prompt, completion, label) where label is a thumbs-up/down bool.
        Useful when you only have one-sided feedback, not pairwise ranks.

Switch between them with --method {dpo,orpo,kto}. DPO/ORPO share data
format; KTO needs a unary dataset.
"""

import argparse
import logging

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    DPOConfig,
    DPOTrainer,
    KTOConfig,
    KTOTrainer,
    ORPOConfig,
    ORPOTrainer,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Per-method default datasets (TRL maintains canonical ones).
DEFAULT_DATASETS = {
    "dpo": "trl-lib/ultrafeedback_binarized",
    "orpo": "trl-lib/ultrafeedback_binarized",
    "kto": "trl-lib/kto-mix-14k",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the method, paths, dataset, LoRA, and training."""
    parser = argparse.ArgumentParser(
        description="Preference-optimization fine-tuning (DPO / ORPO / KTO).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--method",
        type=str,
        default="dpo",
        choices=["dpo", "orpo", "kto"],
        help="Preference-optimization variant to use.",
    )

    # Model / output paths
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output-dir", type=str, default="./preference_finetuned")
    parser.add_argument("--hf-token", type=str, default=None)

    # Dataset
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="HF dataset id. If unset, picks a default for the chosen method.",
    )
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)

    # LoRA (highly recommended — full preference tuning is memory-heavy)
    parser.add_argument(
        "--use-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Training hyperparameters
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Method-specific knobs
    parser.add_argument("--beta", type=float, default=0.1, help="DPO/KTO beta.")
    parser.add_argument(
        "--orpo-beta",
        type=float,
        default=0.1,
        help="ORPO odds-ratio loss weight (lambda in the paper).",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", type=str, default="none")

    return parser.parse_args()


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    """Build a PEFT LoraConfig for causal LM fine-tuning."""
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_trainer(args, model, tokenizer, train_ds, eval_ds, peft_config):
    """Dispatch to the right TRL trainer + config based on --method."""
    common = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        seed=args.seed,
        report_to=args.report_to,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    if args.method == "dpo":
        config = DPOConfig(beta=args.beta, **common)
        # ref_model=None + peft_config -> TRL builds the reference from the
        # frozen base model (no second copy in memory).
        return DPOTrainer(
            model=model,
            ref_model=None,
            args=config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

    if args.method == "orpo":
        # ORPO is reference-free — no ref_model arg.
        config = ORPOConfig(beta=args.orpo_beta, **common)
        return ORPOTrainer(
            model=model,
            args=config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

    # KTO
    config = KTOConfig(beta=args.beta, **common)
    return KTOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )


def main() -> None:
    """End-to-end pipeline: load -> build trainer for chosen method -> train -> save."""
    args = parse_args()
    dataset_name = args.dataset_name or DEFAULT_DATASETS[args.method]
    logger.info("Method: %s | Model: %s | Dataset: %s", args.method, args.model_name, dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=args.hf_token)

    train_ds = load_dataset(dataset_name, split=args.train_split)
    eval_ds = load_dataset(dataset_name, split=args.eval_split) if args.eval_split else None
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))

    peft_config = build_lora_config(args) if args.use_lora else None

    trainer = build_trainer(args, model, tokenizer, train_ds, eval_ds, peft_config)
    logger.info("Starting training (%s).", args.method.upper())
    trainer.train()

    logger.info("Saving to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
