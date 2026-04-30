"""Knowledge distillation: train a small student to mimic a large teacher.

Online logit distillation:
1. Both teacher (frozen) and student forward-pass on the same batch.
2. Loss = alpha * KL(student || teacher) + (1 - alpha) * cross_entropy(labels)
   - Temperature `T` softens both distributions (Hinton et al., 2015).
   - When alpha = 1.0, the student learns purely from the teacher.

Important: teacher and student must share a tokenizer / vocabulary, since
the loss compares per-token logit distributions. Default pair is
Llama-3.1-8B-Instruct (teacher) -> Llama-3.2-1B-Instruct (student); both
share the Llama 3 tokenizer.
"""

import argparse
import logging

import torch
import torch.nn.functional as F
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
    """Parse CLI arguments for teacher/student paths, distillation knobs, and training."""
    parser = argparse.ArgumentParser(
        description="Knowledge distillation: small student mimics large teacher.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Models
    parser.add_argument("--teacher-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--student-model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output-dir", type=str, default="./distilled_student")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument(
        "--teacher-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )

    # Dataset
    parser.add_argument("--dataset-name", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--max-train-samples", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=512)

    # Distillation knobs
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Softmax temperature applied to both teacher and student logits.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight on KL term (1 - alpha goes to standard CE).",
    )

    # Training hyperparameters
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", type=str, default="none")

    return parser.parse_args()


DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class DistillationTrainer(Trainer):
    """Trainer subclass that adds a teacher-student KL term to the loss."""

    def __init__(self, *args, teacher_model, temperature, alpha, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        # Teacher is frozen, eval-mode only.
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Standard student forward pass; HF Trainer puts labels in `inputs`,
        # so the model returns a CE loss over shifted labels for free.
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        student_ce = student_outputs.loss

        # Teacher forward pass — no grad, run on the same device as student.
        with torch.no_grad():
            teacher_inputs = {k: v.to(self.teacher_model.device) for k, v in inputs.items() if k != "labels"}
            teacher_logits = self.teacher_model(**teacher_inputs).logits.to(student_logits.device)

        # KL(student || teacher) at temperature T, on shifted positions.
        # Shift: predict token t+1 from positions [..., t] for both models.
        s = student_logits[..., :-1, :].contiguous()
        t = teacher_logits[..., :-1, :].contiguous()
        s_log_probs = F.log_softmax(s / self.temperature, dim=-1)
        t_probs = F.softmax(t / self.temperature, dim=-1)
        # Hinton et al. scale KL by T^2 so its gradient magnitude is comparable to CE.
        kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (self.temperature ** 2)

        loss = self.alpha * kl + (1.0 - self.alpha) * student_ce
        return (loss, student_outputs) if return_outputs else loss


def load_models_and_tokenizer(args: argparse.Namespace):
    """Load student (trainable) and teacher (frozen). Tokenizer is shared."""
    logger.info("Loading tokenizer from student: %s", args.student_model)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading student: %s", args.student_model)
    student = AutoModelForCausalLM.from_pretrained(args.student_model, token=args.hf_token)

    logger.info("Loading teacher: %s (dtype=%s)", args.teacher_model, args.teacher_dtype)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=DTYPE_MAP[args.teacher_dtype],
        token=args.hf_token,
        device_map="auto",
    )

    if teacher.config.vocab_size != student.config.vocab_size:
        raise ValueError(
            f"Token-level distillation requires identical vocabularies. "
            f"Teacher vocab={teacher.config.vocab_size}, student vocab={student.config.vocab_size}."
        )

    return student, teacher, tokenizer


def prepare_dataset(args: argparse.Namespace, tokenizer):
    """Load + tokenize the dataset. labels = input_ids for causal LM."""
    logger.info("Loading dataset: %s", args.dataset_name)
    raw = (
        load_dataset(args.dataset_name, args.dataset_config)
        if args.dataset_config
        else load_dataset(args.dataset_name)
    )
    train = raw[args.train_split]
    if args.max_train_samples:
        train = train.select(range(min(args.max_train_samples, len(train))))

    def tokenize_fn(examples):
        encodings = tokenizer(
            examples[args.text_field],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        encodings["labels"] = [ids[:] for ids in encodings["input_ids"]]
        return encodings

    return train.map(tokenize_fn, batched=True, remove_columns=train.column_names)


def main() -> None:
    """End-to-end pipeline: load student+teacher -> tokenize -> distill -> save student."""
    args = parse_args()

    student, teacher, tokenizer = load_models_and_tokenizer(args)
    train_ds = prepare_dataset(args, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
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
        bf16=args.bf16,
        seed=args.seed,
        report_to=args.report_to,
    )

    trainer = DistillationTrainer(
        model=student,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        teacher_model=teacher,
        temperature=args.temperature,
        alpha=args.alpha,
    )

    logger.info("Starting distillation (alpha=%.2f, T=%.2f).", args.alpha, args.temperature)
    trainer.train()

    logger.info("Saving distilled student to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
