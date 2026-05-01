"""Preference-optimization (DPO / ORPO / KTO) in pure PyTorch — no TRL.

Same CLI as `train.py`, but the three preference losses are implemented
manually:

* `compute_logprobs`     — per-sequence sum / mean log-prob of response tokens
* `dpo_loss`             — log-sigmoid contrast on policy-vs-ref log-ratios
* `orpo_loss`            — SFT NLL on chosen + odds-ratio contrast
* `kto_loss`             — Kahneman-Tversky utility on unary thumbs-up/down
* `cosine_schedule_lr`   — manual warmup + cosine decay
* `disable_adapter`      — context manager that turns LoRA off so the base
                           model acts as the (frozen) reference

HF Transformers + datasets are used for model / tokenizer / dataset loading.
PEFT/LoRA is kept as an orthogonal option; without LoRA the script loads a
separate frozen reference for DPO/KTO.
"""

import argparse
import logging
import math
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_DATASETS = {
    "dpo": "trl-lib/ultrafeedback_binarized",
    "orpo": "trl-lib/ultrafeedback_binarized",
    "kto": "trl-lib/kto-mix-14k",
}


# ------------------------------------------------------------------
# CLI (matches train.py)
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Same flags as train.py so the two scripts are interchangeable."""
    parser = argparse.ArgumentParser(
        description="Preference optimization (DPO / ORPO / KTO) in pure PyTorch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--method", type=str, default="dpo", choices=["dpo", "orpo", "kto"])

    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output-dir", type=str, default="./preference_finetuned")
    parser.add_argument("--hf-token", type=str, default=None)

    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)

    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine",
                        choices=["cosine", "linear"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--beta", type=float, default=0.1, help="DPO/KTO temperature.")
    parser.add_argument("--orpo-beta", type=float, default=0.1, help="ORPO odds-ratio loss weight.")
    parser.add_argument("--kto-desirable-weight", type=float, default=1.0)
    parser.add_argument("--kto-undesirable-weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ------------------------------------------------------------------
# Tokenization
# ------------------------------------------------------------------

def render_prompt(example: Dict[str, Any], tokenizer) -> str:
    """Pull the prompt out of either a `prompt` field or the chosen[:-1] tail."""
    if "prompt" in example and example["prompt"] is not None:
        prompt = example["prompt"]
        if isinstance(prompt, list):
            return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        return prompt
    # Fallback: derive from chosen by stripping its last (assistant) turn.
    chosen = example["chosen"]
    if isinstance(chosen, list):
        return tokenizer.apply_chat_template(chosen[:-1], tokenize=False, add_generation_prompt=True)
    raise ValueError("Cannot derive prompt without a `prompt` field or a list-format `chosen`.")


def render_response(field: Any, tokenizer) -> str:
    """Render a chosen/rejected/completion field whether it's a string or messages."""
    if isinstance(field, list):
        # Take only the final assistant message — the prompt was rendered separately.
        return field[-1]["content"] if isinstance(field[-1], dict) else str(field[-1])
    return field


def encode_pair(example, tokenizer, max_prompt: int, max_total: int):
    """Return prompt+chosen and prompt+rejected ids, plus the prompt length (for response masking)."""
    prompt_text = render_prompt(example, tokenizer)
    chosen_text = render_response(example["chosen"], tokenizer)
    rejected_text = render_response(example["rejected"], tokenizer)

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)[-max_prompt:]
    chosen_resp = tokenizer.encode(chosen_text, add_special_tokens=False)
    rejected_resp = tokenizer.encode(rejected_text, add_special_tokens=False)

    # Append EOS so the response has a clear terminator.
    if tokenizer.eos_token_id is not None:
        chosen_resp = chosen_resp + [tokenizer.eos_token_id]
        rejected_resp = rejected_resp + [tokenizer.eos_token_id]

    chosen_full = (prompt_ids + chosen_resp)[:max_total]
    rejected_full = (prompt_ids + rejected_resp)[:max_total]
    return {
        "chosen_ids": chosen_full,
        "rejected_ids": rejected_full,
        "prompt_len": len(prompt_ids),
    }


def encode_unary(example, tokenizer, max_prompt: int, max_total: int):
    """KTO: prompt + completion + thumbs-up/down label."""
    prompt_text = render_prompt(example, tokenizer)
    completion_text = render_response(example["completion"], tokenizer)
    label = bool(example["label"])

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)[-max_prompt:]
    resp_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        resp_ids = resp_ids + [tokenizer.eos_token_id]
    full = (prompt_ids + resp_ids)[:max_total]
    return {
        "input_ids": full,
        "prompt_len": len(prompt_ids),
        "label": label,
    }


def pad_and_stack(seqs: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of token-id lists to the same length; build attention masks."""
    max_len = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids, mask


def make_pair_collator(pad_id: int):
    """Collator for DPO/ORPO: stacks chosen + rejected and packs prompt lengths."""
    def collate(batch):
        chosen_ids, chosen_mask = pad_and_stack([s["chosen_ids"] for s in batch], pad_id)
        rejected_ids, rejected_mask = pad_and_stack([s["rejected_ids"] for s in batch], pad_id)
        prompt_lens = torch.tensor([s["prompt_len"] for s in batch], dtype=torch.long)
        return {
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask,
            "prompt_lens": prompt_lens,
        }
    return collate


def make_unary_collator(pad_id: int):
    """Collator for KTO: input_ids + attention_mask + prompt_lens + bool labels."""
    def collate(batch):
        ids, mask = pad_and_stack([s["input_ids"] for s in batch], pad_id)
        prompt_lens = torch.tensor([s["prompt_len"] for s in batch], dtype=torch.long)
        labels = torch.tensor([s["label"] for s in batch], dtype=torch.bool)
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "prompt_lens": prompt_lens,
            "labels": labels,
        }
    return collate


# ------------------------------------------------------------------
# Per-sequence log-prob over response tokens
# ------------------------------------------------------------------

def response_mask(attention_mask: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
    """1 on response token positions (i.e. position >= prompt_len) AND non-padding."""
    B, L = attention_mask.shape
    arange = torch.arange(L, device=attention_mask.device).unsqueeze(0).expand(B, L)
    is_response = arange >= prompt_lens.unsqueeze(1)
    return (is_response & attention_mask.bool()).float()


def compute_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
    average: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sum (or mean) log p(response | prompt). Returns (logp, per-token-logp_for_nll_use)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits

    # Standard causal-LM shift: logits at position t predict token t+1.
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    log_probs_full = F.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs_full.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Restrict to response positions; the shift means position t in `shift_*`
    # corresponds to predicting token at original position t+1, so the response
    # mask shifts by one to the left.
    resp = response_mask(attention_mask, prompt_lens)
    resp = resp[..., 1:]  # align with shift

    masked_logp = token_logp * resp
    seq_sum = masked_logp.sum(dim=-1)
    if average:
        seq_logp = seq_sum / resp.sum(dim=-1).clamp(min=1)
    else:
        seq_logp = seq_sum
    return seq_logp, masked_logp.detach()


# ------------------------------------------------------------------
# Losses
# ------------------------------------------------------------------

def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
):
    """L_DPO = -log σ(β · (Δlogp_chosen - Δlogp_rejected))."""
    chosen_logratio = policy_chosen - ref_chosen
    rejected_logratio = policy_rejected - ref_rejected
    logits = beta * (chosen_logratio - rejected_logratio)
    loss = -F.logsigmoid(logits).mean()
    chosen_reward = (beta * chosen_logratio).detach().mean()
    rejected_reward = (beta * rejected_logratio).detach().mean()
    return loss, {
        "chosen_reward": chosen_reward.item(),
        "rejected_reward": rejected_reward.item(),
        "margin": (chosen_reward - rejected_reward).item(),
    }


def _log1m_exp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 - exp(x)) for x < 0."""
    # See Mächler (2012). Branch on -log(2) for stability.
    threshold = -math.log(2.0)
    return torch.where(
        x > threshold,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


def orpo_loss(
    policy_chosen_avg: torch.Tensor,    # average per-token logp over chosen
    policy_rejected_avg: torch.Tensor,  # average per-token logp over rejected
    chosen_nll: torch.Tensor,           # NLL on chosen (the SFT term)
    beta: float,
):
    """L_ORPO = NLL(chosen) + β · (-log σ(log_odds_chosen - log_odds_rejected))."""
    log_odds_chosen = policy_chosen_avg - _log1m_exp(policy_chosen_avg)
    log_odds_rejected = policy_rejected_avg - _log1m_exp(policy_rejected_avg)
    log_ratio = log_odds_chosen - log_odds_rejected
    or_term = -F.logsigmoid(log_ratio).mean()

    loss = chosen_nll + beta * or_term
    return loss, {
        "sft_nll": chosen_nll.item(),
        "or_term": or_term.item(),
        "log_odds_diff": log_ratio.detach().mean().item(),
    }


def kto_loss(
    policy_logp: torch.Tensor,       # per-sample logp under policy
    ref_logp: torch.Tensor,          # per-sample logp under reference
    labels: torch.Tensor,            # bool tensor
    beta: float,
    desirable_weight: float,
    undesirable_weight: float,
):
    """KTO: σ-shaped utility around a batch-level KL anchor (z_0)."""
    # Per-sample log-ratio (the "implicit reward" r_θ(x, y) = logπ - logπ_ref).
    log_ratio = policy_logp - ref_logp.detach()

    # Batch-level KL anchor — clamp to ≥ 0 per the paper.
    desirable_mask = labels
    undesirable_mask = ~labels

    # Use unmatched-class mean as the reference KL (TRL convention): it works
    # even when a mini-batch is all-positive or all-negative because we fall
    # back to the same-class mean.
    if undesirable_mask.any():
        z0_des = log_ratio[undesirable_mask].mean().detach().clamp(min=0)
    else:
        z0_des = log_ratio.mean().detach().clamp(min=0)
    if desirable_mask.any():
        z0_un = log_ratio[desirable_mask].mean().detach().clamp(min=0)
    else:
        z0_un = log_ratio.mean().detach().clamp(min=0)

    losses = torch.zeros_like(log_ratio)
    if desirable_mask.any():
        losses[desirable_mask] = desirable_weight * (1.0 - torch.sigmoid(beta * (log_ratio[desirable_mask] - z0_des)))
    if undesirable_mask.any():
        losses[undesirable_mask] = undesirable_weight * (1.0 - torch.sigmoid(beta * (z0_un - log_ratio[undesirable_mask])))

    loss = losses.mean()
    return loss, {
        "desirable_count": int(desirable_mask.sum().item()),
        "undesirable_count": int(undesirable_mask.sum().item()),
        "kl_anchor": float(z0_des.item()),
        "log_ratio_mean": float(log_ratio.detach().mean().item()),
    }


# ------------------------------------------------------------------
# Reference-model handling (PEFT-aware)
# ------------------------------------------------------------------

@contextmanager
def disable_adapter(model):
    """Toggle off LoRA adapters so the same module behaves as the frozen base."""
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        yield


def ref_logprobs(method: str, model, ref_model, input_ids, attention_mask, prompt_lens, average=False):
    """Compute logp under the reference. With LoRA, reuse `model` with adapters off."""
    if ref_model is not None:
        with torch.no_grad():
            logp, _ = compute_logprobs(ref_model, input_ids, attention_mask, prompt_lens, average=average)
        return logp
    with torch.no_grad(), disable_adapter(model):
        logp, _ = compute_logprobs(model, input_ids, attention_mask, prompt_lens, average=average)
    return logp


# ------------------------------------------------------------------
# LR schedule
# ------------------------------------------------------------------

def schedule_lr(step: int, warmup_steps: int, total_steps: int, kind: str) -> float:
    """Manual warmup + cosine (or linear) decay multiplier in [0, 1]."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    if kind == "linear":
        return max(0.0, 1.0 - progress)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ------------------------------------------------------------------
# Setup helpers
# ------------------------------------------------------------------

def load_models_and_tokenizer(args):
    """Tokenizer + policy (with optional LoRA) + frozen reference (skipped if LoRA on)."""
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    common = dict(token=args.hf_token, torch_dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading policy: %s", args.model_name)
    policy = AutoModelForCausalLM.from_pretrained(args.model_name, **common)

    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy = get_peft_model(policy, peft_config)
        policy.print_trainable_parameters()

    # ORPO is reference-free; otherwise we either reuse the base under LoRA
    # (disable_adapter context) or load a separate frozen copy.
    ref_model = None
    if args.method != "orpo" and not args.use_lora:
        logger.info("Loading frozen reference policy.")
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, **common)
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()

    return policy, ref_model, tokenizer


def build_dataset(args, tokenizer):
    """Load + tokenize per-method; returns a torch Dataset and the right collator."""
    name = args.dataset_name or DEFAULT_DATASETS[args.method]
    logger.info("Loading dataset: %s", name)

    raw = load_dataset(name, split=args.train_split)
    if args.max_train_samples:
        raw = raw.select(range(min(args.max_train_samples, len(raw))))

    if args.method == "kto":
        encoded = raw.map(
            lambda ex: encode_unary(ex, tokenizer, args.max_prompt_length, args.max_length),
            remove_columns=raw.column_names,
        )
        collator = make_unary_collator(tokenizer.pad_token_id)
    else:
        encoded = raw.map(
            lambda ex: encode_pair(ex, tokenizer, args.max_prompt_length, args.max_length),
            remove_columns=raw.column_names,
        )
        collator = make_pair_collator(tokenizer.pad_token_id)

    return encoded, collator


# ------------------------------------------------------------------
# Per-method training step
# ------------------------------------------------------------------

def step_dpo(policy, ref_model, batch, args, device):
    """One forward pass over chosen + rejected, plus reference logp; returns loss + metrics."""
    chosen_ids = batch["chosen_ids"].to(device)
    chosen_mask = batch["chosen_mask"].to(device)
    rejected_ids = batch["rejected_ids"].to(device)
    rejected_mask = batch["rejected_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)

    policy_chosen, _ = compute_logprobs(policy, chosen_ids, chosen_mask, prompt_lens)
    policy_rejected, _ = compute_logprobs(policy, rejected_ids, rejected_mask, prompt_lens)

    ref_chosen = ref_logprobs(args.method, policy, ref_model, chosen_ids, chosen_mask, prompt_lens)
    ref_rejected = ref_logprobs(args.method, policy, ref_model, rejected_ids, rejected_mask, prompt_lens)

    return dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, args.beta)


def step_orpo(policy, batch, args, device):
    """ORPO: NLL on chosen + odds-ratio contrast. No reference model needed."""
    chosen_ids = batch["chosen_ids"].to(device)
    chosen_mask = batch["chosen_mask"].to(device)
    rejected_ids = batch["rejected_ids"].to(device)
    rejected_mask = batch["rejected_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)

    # Average per-token logp on both halves of the pair.
    policy_chosen_avg, _ = compute_logprobs(policy, chosen_ids, chosen_mask, prompt_lens, average=True)
    policy_rejected_avg, _ = compute_logprobs(policy, rejected_ids, rejected_mask, prompt_lens, average=True)

    # The SFT NLL is the negation of the average chosen logp.
    chosen_nll = -policy_chosen_avg.mean()

    return orpo_loss(policy_chosen_avg, policy_rejected_avg, chosen_nll, args.orpo_beta)


def step_kto(policy, ref_model, batch, args, device):
    """KTO: unary thumbs-up/down. Needs a reference (or LoRA-disabled base)."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)
    labels = batch["labels"].to(device)

    policy_logp, _ = compute_logprobs(policy, input_ids, attention_mask, prompt_lens)
    ref_logp = ref_logprobs(args.method, policy, ref_model, input_ids, attention_mask, prompt_lens)

    return kto_loss(
        policy_logp, ref_logp, labels,
        beta=args.beta,
        desirable_weight=args.kto_desirable_weight,
        undesirable_weight=args.kto_undesirable_weight,
    )


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train_loop(policy, ref_model, dataset, collator, args, device):
    """Manual loop: AMP autocast (bf16) + grad accumulation + LR schedule + save."""
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = max(1, int(steps_per_epoch * args.num_train_epochs))
    warmup_steps = int(total_steps * args.warmup_ratio)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    autocast_kwargs = (
        dict(enabled=torch.cuda.is_available(), dtype=torch.bfloat16)
        if args.bf16 else dict(enabled=False)
    )

    logger.info(
        "Training: %d optimizer steps (%d warmup), grad_accum=%d, method=%s.",
        total_steps, warmup_steps, args.gradient_accumulation_steps, args.method,
    )

    policy.train()
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    epochs_to_run = max(1, math.ceil(args.num_train_epochs))
    done = False
    for epoch in range(epochs_to_run):
        for batch in loader:
            if done:
                break
            with torch.cuda.amp.autocast(**autocast_kwargs):
                if args.method == "dpo":
                    loss, stats = step_dpo(policy, ref_model, batch, args, device)
                elif args.method == "orpo":
                    loss, stats = step_orpo(policy, batch, args, device)
                else:
                    loss, stats = step_kto(policy, ref_model, batch, args, device)
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            running_loss += loss.item()
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps != 0:
                continue

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)

            lr_scale = schedule_lr(global_step, warmup_steps, total_steps, args.lr_scheduler_type)
            for pg in optimizer.param_groups:
                pg["lr"] = args.learning_rate * lr_scale

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.logging_steps == 0:
                avg = running_loss / args.logging_steps
                running_loss = 0.0
                logger.info(
                    "epoch=%d step=%d loss=%.4f lr=%.2e %s",
                    epoch, global_step, avg, args.learning_rate * lr_scale, stats,
                )

            if global_step % args.save_steps == 0:
                save_policy(policy, None, args.output_dir, step=global_step)

            if global_step >= total_steps:
                done = True
                break

        if done:
            break

    return global_step


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

def save_policy(policy, tokenizer, output_dir: str, step: Optional[int] = None):
    """Save adapters (or full model) to output_dir / output_dir/checkpoint-{step}."""
    target = Path(output_dir)
    if step is not None:
        target = target / f"checkpoint-{step}"
    target.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(target)
    if tokenizer is not None:
        tokenizer.save_pretrained(target)
    logger.info("Saved to %s", target)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.method == "orpo" and args.use_lora is False and not torch.cuda.is_available():
        logger.warning("ORPO without LoRA on CPU will be very slow.")

    policy, ref_model, tokenizer = load_models_and_tokenizer(args)
    policy.to(device)
    if ref_model is not None:
        ref_model.to(device)

    dataset, collator = build_dataset(args, tokenizer)

    final_step = train_loop(policy, ref_model, dataset, collator, args, device)

    save_policy(policy, tokenizer, args.output_dir)
    logger.info("Done. Trained for %d optimizer steps with method=%s.", final_step, args.method)


if __name__ == "__main__":
    main()
