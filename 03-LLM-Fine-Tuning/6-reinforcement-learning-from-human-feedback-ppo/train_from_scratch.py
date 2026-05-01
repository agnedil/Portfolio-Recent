"""RLHF / PPO implementation in pure PyTorch — no TRL dependency.

Same goal and CLI shape as `train.py`, but every PPO-specific component is
written from scratch:

* `ValueHead`, `PolicyWithValueHead`     — the policy + critic
* `compute_logprobs`, `gather_logprobs`  — per-token log-prob extraction
* `rollout`                              — sample, generate, score
* `compute_token_rewards`                — reward model + per-token KL
* `compute_gae`                          — GAE-λ advantages and returns
* `ppo_loss`                             — clip / adaptive-KL variants
* `AdaptiveKLController`                 — βₜ update rule for `--method kl`
* `train_loop`                           — collect → score → update → log

HF Transformers is still used for the underlying LM + reward classifier
(implementing those from scratch is a different exercise). PEFT / LoRA and
bitsandbytes quantization are kept as orthogonal options; the PPO algorithm
itself is implemented manually.
"""


import argparse
import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """CLI mirrors `train.py` so the two scripts are interchangeable."""
    parser = argparse.ArgumentParser(
        description="RLHF / PPO implemented in pure PyTorch (no TRL).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--method", type=str, default="clip", choices=["clip", "kl"])
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output-dir", type=str, default="./ppo_finetuned_scratch")
    parser.add_argument("--hf-token", type=str, default=None)

    parser.add_argument(
        "--reward-model",
        type=str,
        default="OpenAssistant/reward-model-deberta-v3-large-v2",
    )

    parser.add_argument("--dataset-name", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--prompt-field", type=str, default="instruction")
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-prompt-length", type=int, default=256)

    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gen-temperature", type=float, default=0.9)
    parser.add_argument("--gen-top-p", type=float, default=0.95)

    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-in-8bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bnb-4bit-quant-type", type=str, default="nf4", choices=["nf4", "fp4"])

    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--learning-rate", type=float, default=1.41e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mini-batch-size", type=int, default=2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--cliprange-disabled", type=float, default=100.0)
    parser.add_argument("--init-kl-coef", type=float, default=0.05,
                        help="KL coefficient used for the per-token reward shaping (vs reference).")
    parser.add_argument("--target-kl", type=float, default=6.0,
                        help="Target KL for adaptive control (vs old policy) in --method kl.")
    parser.add_argument("--vf-coef", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ------------------------------------------------------------------
# Model wrapper
# ------------------------------------------------------------------

class ValueHead(nn.Module):
    """Single linear layer from final hidden state to a scalar value per token."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.summary = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.summary(hidden_states).squeeze(-1)


class PolicyWithValueHead(nn.Module):
    """Wrap a HF causal LM with a value head; reuse base model for the LM forward."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        hidden_size = base_model.config.hidden_size
        self.value_head = ValueHead(hidden_size).to(
            dtype=base_model.dtype, device=base_model.device
        )

    def forward(self, input_ids, attention_mask=None):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        values = self.value_head(out.hidden_states[-1].to(self.value_head.summary.weight.dtype))
        return out.logits, values

    def generate(self, *args, **kwargs):
        return self.base.generate(*args, **kwargs)


# ------------------------------------------------------------------
# Adaptive KL controller (for --method kl)
# ------------------------------------------------------------------

class AdaptiveKLController:
    """Multiply β by 1.5 / divide by 1.5 to track a target KL after each batch."""

    def __init__(self, init_kl_coef: float, target: float):
        self.value = init_kl_coef
        self.target = target

    def update(self, current_kl: float) -> None:
        proportional_error = (current_kl / self.target) - 1.0
        # Clamp to keep multiplicative updates within [1/1.5, 1.5]
        mult = 1.0 + max(min(0.5, proportional_error), -0.333)
        self.value *= mult


# ------------------------------------------------------------------
# Per-token utilities
# ------------------------------------------------------------------

def gather_logprobs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """log p(target_ids | context) for each position."""
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)


def compute_logprobs_for_response(
    model: PolicyWithValueHead,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start: int,
):
    """Forward pass on the concatenated (prompt + response) sequence; return per-response-token logp + values."""
    logits, values = model(full_ids, attention_mask=attention_mask)
    # Shift: logits at position t predict token t+1.
    shift_logits = logits[:, response_start - 1:-1, :]
    shift_targets = full_ids[:, response_start:]
    logp = gather_logprobs(shift_logits, shift_targets)
    # Values are aligned with response token positions.
    response_values = values[:, response_start - 1:-1]
    return logp, response_values


@torch.no_grad()
def compute_ref_logprobs_for_response(
    ref_model,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start: int,
) -> torch.Tensor:
    """Same as above but for the frozen reference; no value head."""
    out = ref_model(input_ids=full_ids, attention_mask=attention_mask, return_dict=True)
    shift_logits = out.logits[:, response_start - 1:-1, :]
    shift_targets = full_ids[:, response_start:]
    return gather_logprobs(shift_logits, shift_targets)


# ------------------------------------------------------------------
# Reward shaping + GAE
# ------------------------------------------------------------------

def compute_token_rewards(
    scalar_rewards: torch.Tensor,    # [B]
    logp: torch.Tensor,              # [B, L]
    ref_logp: torch.Tensor,          # [B, L]
    response_mask: torch.Tensor,     # [B, L]
    kl_ref_coef: float,
) -> torch.Tensor:
    """Per-token reward = -β·KL(policy || ref) + scalar at the last response token."""
    kl = logp - ref_logp                       # token-level KL estimate
    rewards = -kl_ref_coef * kl
    rewards = rewards * response_mask          # zero-out padding positions

    # Place the scalar reward on the last *valid* token of each response.
    last_token_idx = response_mask.sum(dim=1).long() - 1
    last_token_idx = last_token_idx.clamp(min=0)
    rewards[torch.arange(rewards.size(0), device=rewards.device), last_token_idx] += scalar_rewards
    return rewards


def compute_gae(
    rewards: torch.Tensor,           # [B, L]
    values: torch.Tensor,            # [B, L]
    response_mask: torch.Tensor,     # [B, L]
    gamma: float,
    lam: float,
):
    """Generalized advantage estimation, masked over padding tokens."""
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.size(0), device=rewards.device)
    L = rewards.size(1)
    for t in reversed(range(L)):
        next_value = values[:, t + 1] if t + 1 < L else torch.zeros_like(values[:, 0])
        next_mask = response_mask[:, t + 1] if t + 1 < L else torch.zeros_like(response_mask[:, 0])
        delta = rewards[:, t] + gamma * next_value * next_mask - values[:, t]
        last_gae = delta + gamma * lam * next_mask * last_gae
        advantages[:, t] = last_gae
    returns = advantages + values
    return advantages * response_mask, returns * response_mask


# ------------------------------------------------------------------
# PPO loss
# ------------------------------------------------------------------

@dataclass
class PPOStats:
    policy_loss: float
    value_loss: float
    kl: float
    clip_frac: float


def ppo_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    new_values: torch.Tensor,
    response_mask: torch.Tensor,
    method: str,
    cliprange: float,
    kl_coef: float,
    vf_coef: float,
):
    """PPO-Clip and adaptive-KL share most of the code; the surrogate differs."""
    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)

    # k3 KL estimator (Schulman blog) — low-variance, always non-negative.
    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio)
    approx_kl = (approx_kl * response_mask).sum() / response_mask.sum().clamp(min=1)

    if method == "clip":
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantages
        per_token_surr = -torch.min(unclipped, clipped)
        clip_frac = ((torch.abs(ratio - 1) > cliprange).float() * response_mask).sum()
        clip_frac = clip_frac / response_mask.sum().clamp(min=1)
        policy_loss = (per_token_surr * response_mask).sum() / response_mask.sum().clamp(min=1)
    else:
        # PPO-KL: importance-weighted advantage minus β·KL term.
        per_token_surr = -ratio * advantages
        policy_loss = (per_token_surr * response_mask).sum() / response_mask.sum().clamp(min=1)
        policy_loss = policy_loss + kl_coef * approx_kl
        clip_frac = torch.tensor(0.0, device=policy_loss.device)

    value_error = (new_values - returns).pow(2)
    value_loss = (value_error * response_mask).sum() / response_mask.sum().clamp(min=1)

    total = policy_loss + vf_coef * value_loss
    return total, PPOStats(
        policy_loss=policy_loss.item(),
        value_loss=value_loss.item(),
        kl=approx_kl.item(),
        clip_frac=clip_frac.item(),
    )


# ------------------------------------------------------------------
# Loading / setup
# ------------------------------------------------------------------

def build_quantization_config(args):
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Pass only one of --load-in-4bit / --load-in-8bit.")
    if args.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if args.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_models_and_tokenizer(args):
    bnb_config = build_quantization_config(args)
    common = dict(token=args.hf_token, torch_dtype=torch.bfloat16)
    if bnb_config is not None:
        common["quantization_config"] = bnb_config
        common["device_map"] = "auto"

    logger.info("Loading base policy: %s", args.model_name)
    base = AutoModelForCausalLM.from_pretrained(args.model_name, **common)

    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base = get_peft_model(base, peft_config)
        base.print_trainable_parameters()

    policy = PolicyWithValueHead(base)

    logger.info("Loading frozen reference policy.")
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, **common)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for generation

    return policy, ref_model, tokenizer


def prepare_dataloader(args, tokenizer):
    raw = load_dataset(args.dataset_name, split=args.dataset_split)
    if args.max_train_samples:
        raw = raw.select(range(min(args.max_train_samples, len(raw))))

    def encode(sample):
        ids = tokenizer.encode(sample[args.prompt_field], truncation=True,
                               max_length=args.max_prompt_length)
        return {"input_ids": ids, "query": tokenizer.decode(ids, skip_special_tokens=True)}

    raw = raw.map(encode, remove_columns=raw.column_names)

    def collate(batch):
        ids = [torch.tensor(s["input_ids"]) for s in batch]
        max_len = max(t.size(0) for t in ids)
        padded = torch.full((len(ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attn = torch.zeros_like(padded)
        for i, t in enumerate(ids):
            padded[i, -t.size(0):] = t
            attn[i, -t.size(0):] = 1
        return {
            "input_ids": padded,
            "attention_mask": attn,
            "query": [s["query"] for s in batch],
        }

    return DataLoader(raw, batch_size=args.batch_size, shuffle=True, collate_fn=collate)


# ------------------------------------------------------------------
# Rollout
# ------------------------------------------------------------------

@torch.no_grad()
def rollout(
    policy: PolicyWithValueHead,
    ref_model,
    reward_pipe,
    tokenizer,
    batch,
    args,
    device,
):
    """Generate responses, compute logp/values under policy and ref, score with reward model."""
    policy.eval()

    prompt_ids = batch["input_ids"].to(device)
    prompt_mask = batch["attention_mask"].to(device)
    prompt_len = prompt_ids.size(1)

    # 1. Generate responses.
    gen_out = policy.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    full_ids = gen_out  # [B, prompt_len + L_resp]
    response_ids = full_ids[:, prompt_len:]
    full_mask = (full_ids != tokenizer.pad_token_id).long()
    full_mask[:, :prompt_len] = prompt_mask  # preserve original prompt mask
    response_mask = full_mask[:, prompt_len:].float()

    # 2. Per-token logp + values under the (current) policy.
    logp, values = compute_logprobs_for_response(policy, full_ids, full_mask, prompt_len)

    # 3. Same under the frozen reference.
    ref_logp = compute_ref_logprobs_for_response(ref_model, full_ids, full_mask, prompt_len)

    # 4. Reward model score for each (prompt + response).
    decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in full_ids]
    pipe_out = reward_pipe(decoded, truncation=True)
    scores = torch.tensor([o["score"] for o in pipe_out], dtype=torch.float32, device=device)

    return {
        "full_ids": full_ids,
        "full_mask": full_mask,
        "response_mask": response_mask,
        "old_logp": logp.detach(),
        "ref_logp": ref_logp.detach(),
        "old_values": values.detach(),
        "scores": scores,
        "response_ids": response_ids,
    }


# ------------------------------------------------------------------
# PPO update over the rollout
# ------------------------------------------------------------------

def ppo_update(policy, optimizer, rollout_data, advantages, returns, args, kl_coef):
    """Multiple mini-batch passes (`--ppo-epochs`) over the same rollout."""
    policy.train()

    full_ids = rollout_data["full_ids"]
    full_mask = rollout_data["full_mask"]
    response_mask = rollout_data["response_mask"]
    old_logp = rollout_data["old_logp"]
    prompt_len = full_ids.size(1) - response_mask.size(1)

    # Normalize advantages over the masked entries (standard PPO trick).
    adv_mean = (advantages * response_mask).sum() / response_mask.sum().clamp(min=1)
    adv_var = (((advantages - adv_mean) ** 2) * response_mask).sum() / response_mask.sum().clamp(min=1)
    advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)

    B = full_ids.size(0)
    indices = list(range(B))
    last_stats = None

    for _ in range(args.ppo_epochs):
        # Shuffle and split into mini-batches.
        torch.manual_seed(torch.randint(0, 2**31, (1,)).item())
        perm = torch.randperm(B).tolist()
        for start in range(0, B, args.mini_batch_size):
            mb = perm[start : start + args.mini_batch_size]
            new_logp, new_values = compute_logprobs_for_response(
                policy, full_ids[mb], full_mask[mb], prompt_len
            )
            loss, stats = ppo_loss(
                new_logp=new_logp,
                old_logp=old_logp[mb],
                advantages=advantages[mb],
                returns=returns[mb],
                new_values=new_values,
                response_mask=response_mask[mb],
                method=args.method,
                cliprange=args.cliprange if args.method == "clip" else args.cliprange_disabled,
                kl_coef=kl_coef,
                vf_coef=args.vf_coef,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            last_stats = stats

    return last_stats


# ------------------------------------------------------------------
# Save (LoRA-aware)
# ------------------------------------------------------------------

def save_policy(policy: PolicyWithValueHead, tokenizer, output_dir: str):
    """Save base/adapters + tokenizer + value head separately."""
    base = policy.base
    if hasattr(base, "save_pretrained"):
        base.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(policy.value_head.state_dict(), f"{output_dir}/value_head.pt")
    logger.info("Saved policy (+ value_head.pt) to %s", output_dir)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy, ref_model, tokenizer = load_models_and_tokenizer(args)
    if not args.load_in_4bit and not args.load_in_8bit:
        policy.to(device)
        ref_model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    dataloader = prepare_dataloader(args, tokenizer)

    logger.info("Loading reward model: %s", args.reward_model)
    reward_pipe = pipeline(
        "text-classification",
        model=args.reward_model,
        device=0 if torch.cuda.is_available() else -1,
        token=args.hf_token,
        function_to_apply="none",
    )

    kl_controller = (
        AdaptiveKLController(args.init_kl_coef, args.target_kl)
        if args.method == "kl"
        else None
    )
    # KL coefficient for the per-token *reward shaping* (vs reference) is
    # always fixed at args.init_kl_coef. The adaptive coefficient (vs old
    # policy, in --method kl) is a separate quantity, owned by the controller.
    kl_ref_coef = args.init_kl_coef

    logger.info("Starting PPO (method=%s, max_steps=%d).", args.method, args.max_steps)
    step = 0
    while step < args.max_steps:
        for batch in dataloader:
            if step >= args.max_steps:
                break

            # ---- Collect rollout ----
            data = rollout(policy, ref_model, reward_pipe, tokenizer, batch, args, device)

            token_rewards = compute_token_rewards(
                scalar_rewards=data["scores"],
                logp=data["old_logp"],
                ref_logp=data["ref_logp"],
                response_mask=data["response_mask"],
                kl_ref_coef=kl_ref_coef,
            )
            advantages, returns = compute_gae(
                rewards=token_rewards,
                values=data["old_values"],
                response_mask=data["response_mask"],
                gamma=args.gamma,
                lam=args.gae_lambda,
            )

            # ---- PPO update ----
            kl_coef_for_loss = kl_controller.value if kl_controller else 0.0
            stats = ppo_update(policy, optimizer, data, advantages, returns, args, kl_coef_for_loss)

            # ---- Adaptive KL update (only --method kl) ----
            if kl_controller is not None and stats is not None:
                kl_controller.update(stats.kl)

            mean_r = data["scores"].mean().item()
            if step % 5 == 0:
                logger.info(
                    "step=%d mean_reward=%.4f policy_loss=%.4f value_loss=%.4f kl=%.4f clip_frac=%.3f beta=%.4f",
                    step,
                    mean_r,
                    stats.policy_loss if stats else float("nan"),
                    stats.value_loss if stats else float("nan"),
                    stats.kl if stats else float("nan"),
                    stats.clip_frac if stats else 0.0,
                    kl_controller.value if kl_controller else 0.0,
                )
            step += 1

    save_policy(policy, tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
