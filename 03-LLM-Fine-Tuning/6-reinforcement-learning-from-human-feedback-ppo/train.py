"""RLHF training with Proximal Policy Optimization (PPO).

Three components run together in PPO:
* Policy        — the trainable model (with a value head).
* Reference     — a frozen copy of the policy used for KL regularization.
* Reward model  — a separately trained classifier that scores responses.

Each step:
  1. Sample prompts from the dataset.
  2. Generate responses with the policy.
  3. Score the (prompt, response) pairs with the reward model.
  4. Update the policy with PPO using those rewards, regularized so it
     doesn't drift far from the reference.

PPO has two classic flavours (Schulman et al., 2017). Switch with
`--method`:

* `clip` — PPO-Clip: clips the importance-weighted ratio
           min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t).
           A small fixed KL penalty is kept just for stability.
* `kl`   — PPO-KL: clipping is effectively disabled; an *adaptive* KL
           coefficient is the main constraint, retargeted toward
           `--target-kl` after each update.

Quantization (`--load-in-4bit` / `--load-in-8bit`) is recommended — three
models in VRAM at once is heavy without it. LoRA (default on) further cuts
memory by only training adapter weights.
"""

import argparse
import logging

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the policy, reward model, dataset, PPO knobs, and quantization."""
    parser = argparse.ArgumentParser(
        description="RLHF training with PPO (clip or adaptive-KL variant).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--method",
        type=str,
        default="clip",
        choices=["clip", "kl"],
        help="PPO variant: 'clip' (PPO-Clip) or 'kl' (adaptive KL penalty).",
    )

    # Policy / output paths
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output-dir", type=str, default="./ppo_finetuned")
    parser.add_argument("--hf-token", type=str, default=None)

    # Reward model
    parser.add_argument(
        "--reward-model",
        type=str,
        default="OpenAssistant/reward-model-deberta-v3-large-v2",
        help="HF text-classification model that scores (prompt, response) quality.",
    )

    # Dataset
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="tatsu-lab/alpaca",
        help="Prompts dataset. Only the prompt field is used.",
    )
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument(
        "--prompt-field",
        type=str,
        default="instruction",
        help="Field to use as the prompt (e.g. 'instruction' for Alpaca).",
    )
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-prompt-length", type=int, default=256)

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gen-temperature", type=float, default=0.9)
    parser.add_argument("--gen-top-p", type=float, default=0.95)

    # Quantization (only one should be set)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--load-in-8bit",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--bnb-4bit-quant-type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
    )

    # LoRA — strongly recommended for PPO (3 models in VRAM)
    parser.add_argument(
        "--use-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=1.41e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mini-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--cliprange", type=float, default=0.2,
                        help="PPO-Clip range (used in --method clip).")
    parser.add_argument("--cliprange-disabled", type=float, default=100.0,
                        help="Effective 'no clipping' value used in --method kl.")
    parser.add_argument("--init-kl-coef", type=float, default=0.05,
                        help="Initial KL penalty coefficient.")
    parser.add_argument("--target-kl", type=float, default=6.0,
                        help="Target KL for adaptive control (--method kl).")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", type=str, default="none")

    return parser.parse_args()


def build_quantization_config(args: argparse.Namespace):
    """Return a `BitsAndBytesConfig` if quantization is requested, else None."""
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


def build_ppo_config(args: argparse.Namespace) -> PPOConfig:
    """Translate `--method` into the right PPOConfig knobs."""
    common = dict(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        log_with=None if args.report_to == "none" else args.report_to,
    )

    if args.method == "clip":
        # PPO-Clip: clipping is the main constraint; small fixed KL for stability.
        return PPOConfig(
            cliprange=args.cliprange,
            cliprange_value=args.cliprange,
            init_kl_coef=args.init_kl_coef,
            adap_kl_ctrl=False,
            **common,
        )

    # PPO-KL: clipping effectively disabled; adaptive KL is the main constraint.
    return PPOConfig(
        cliprange=args.cliprange_disabled,
        cliprange_value=args.cliprange_disabled,
        init_kl_coef=args.init_kl_coef,
        adap_kl_ctrl=True,
        target=args.target_kl,
        **common,
    )


def load_models_and_tokenizer(args: argparse.Namespace):
    """Load policy (with value head), frozen reference, and tokenizer."""
    bnb_config = build_quantization_config(args)
    load_kwargs = dict(
        token=args.hf_token,
        torch_dtype=torch.bfloat16,
    )
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    logger.info("Loading policy with value head: %s", args.model_name)
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name,
        peft_config=peft_config,
        **load_kwargs,
    )

    # When LoRA is active, TRL constructs the reference by disabling adapters
    # on the same base — no second model in memory. Otherwise we load a copy.
    ref_policy = None
    if peft_config is None:
        logger.info("Loading frozen reference policy.")
        ref_policy = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model_name, **load_kwargs
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return policy, ref_policy, tokenizer


def prepare_dataset(args: argparse.Namespace, tokenizer):
    """Tokenize prompts; PPO uses prompt-only data and generates responses online."""
    logger.info("Loading dataset: %s", args.dataset_name)
    raw = load_dataset(args.dataset_name, split=args.dataset_split)
    if args.max_train_samples:
        raw = raw.select(range(min(args.max_train_samples, len(raw))))

    def tokenize_fn(sample):
        text = sample[args.prompt_field]
        ids = tokenizer.encode(
            text, truncation=True, max_length=args.max_prompt_length
        )
        return {"input_ids": ids, "query": tokenizer.decode(ids)}

    tokenized = raw.map(tokenize_fn, remove_columns=raw.column_names)
    tokenized.set_format(type="torch")
    return tokenized


def collator(batch):
    """PPO expects a dict-of-lists, not a list-of-dicts."""
    return {key: [sample[key] for sample in batch] for key in batch[0]}


def main() -> None:
    """End-to-end PPO loop: sample -> generate -> score -> update."""
    args = parse_args()
    logger.info("PPO method: %s", args.method)

    policy, ref_policy, tokenizer = load_models_and_tokenizer(args)
    dataset = prepare_dataset(args, tokenizer)
    ppo_config = build_ppo_config(args)

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        ref_model=ref_policy,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    logger.info("Loading reward model: %s", args.reward_model)
    reward_pipe = pipeline(
        "text-classification",
        model=args.reward_model,
        device=ppo_trainer.accelerator.device,
        token=args.hf_token,
        function_to_apply="none",
    )

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    logger.info("Starting PPO training (max_steps=%d).", args.max_steps)
    for step, batch in enumerate(ppo_trainer.dataloader):
        if step >= args.max_steps:
            break

        query_tensors = batch["input_ids"]

        # 1. Generate responses from the current policy.
        response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, **gen_kwargs
        )
        batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

        # 2. Score (prompt + response) pairs with the reward model.
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_out = reward_pipe(texts, truncation=True)
        rewards = [torch.tensor(o["score"], dtype=torch.float32) for o in pipe_out]

        # 3. PPO update — clipping or adaptive-KL behavior comes from PPOConfig.
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if step % 10 == 0:
            mean_r = sum(r.item() for r in rewards) / max(len(rewards), 1)
            logger.info("step=%d mean_reward=%.4f kl=%.4f", step, mean_r,
                        stats.get("objective/kl", float("nan")))

    logger.info("Saving policy + tokenizer to %s", args.output_dir)
    ppo_trainer.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
