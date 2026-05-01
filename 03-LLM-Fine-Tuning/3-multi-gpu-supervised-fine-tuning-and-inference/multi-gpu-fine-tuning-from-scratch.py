"""Multi-GPU fine-tuning of a causal LM — pure PyTorch, no HF Trainer.

Same CLI as `multi-gpu-fine-tuning.py` and the same launch pattern:
    torchrun --nproc_per_node=4 multi-gpu-fine-tuning-from-scratch.py
    torchrun --nproc_per_node=4 multi-gpu-fine-tuning-from-scratch.py --fsdp

Every distributed-training piece is built manually:

* `setup_distributed`            — init NCCL process group from torchrun env
* `wrap_for_distributed`         — DDP or FSDP wrap with auto-wrap policy
* `make_distributed_dataloader`  — DistributedSampler + collate
* `linear_schedule_lr`           — manual warmup + linear decay
* `causal_lm_loss`               — shifted cross-entropy
* `train_loop`                   — grad-accum + AMP + cross-rank reductions
* `evaluate`                     — eval loop with all-reduced loss
* `save_model_distributed`       — handles both DDP unwrap and FSDP gather

Single-process fallback: if `WORLD_SIZE` isn't set (i.e. you ran with plain
`python ...` instead of `torchrun`), the script trains on one GPU without
DDP/FSDP. Useful for smoke-testing on a workstation.
"""

import argparse
import functools
import logging
import math
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch.distributed.fsdp import (
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# CLI (matches multi-gpu-fine-tuning.py)
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Same flags as the HF-Trainer version, plus a few from-scratch knobs."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU fine-tuning in pure PyTorch (DDP / FSDP via torchrun).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--hf-token", type=str, default=None)

    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="validation")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--max-length", type=int, default=512)

    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--evaluation-strategy", type=str, default="steps",
                        choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--fsdp", action=argparse.BooleanOptionalAction, default=False,
                        help="Use FSDP (set when the model doesn't fit on one GPU).")
    parser.add_argument("--fsdp-transformer-layer", type=str, default="LlamaDecoderLayer",
                        help="Decoder-layer class name to wrap with FSDP.")
    parser.add_argument("--fsdp-cpu-offload", action=argparse.BooleanOptionalAction, default=False,
                        help="Offload sharded params/grads to CPU (extra savings, slower).")
    parser.add_argument("--ddp-find-unused-parameters", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ------------------------------------------------------------------
# Distributed setup
# ------------------------------------------------------------------

def setup_distributed():
    """Init NCCL from torchrun env vars; gracefully fall back to single-process."""
    if "WORLD_SIZE" not in os.environ:
        # Single-process: pretend we're a 1-rank world.
        return 0, 1, 0, False

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank, True


def cleanup_distributed(distributed: bool):
    """Drop the NCCL process group cleanly."""
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def configure_logging(rank: int):
    """Only rank 0 logs to keep the console readable."""
    level = logging.INFO if is_main_process(rank) else logging.ERROR
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s - rank{rank} - %(levelname)s - %(message)s",
    )


# ------------------------------------------------------------------
# DDP / FSDP wrapping
# ------------------------------------------------------------------

def find_layer_class(model: torch.nn.Module, class_name: str):
    """Walk module tree to find the class with the requested __name__ (FSDP wrap policy needs the class)."""
    for module in model.modules():
        if module.__class__.__name__ == class_name:
            return module.__class__
    raise ValueError(
        f"No module of class {class_name!r} found. "
        f"Pass --fsdp-transformer-layer with the correct decoder-layer class name."
    )


def build_mp_policy(args):
    """FSDP MixedPrecision config — equivalent to autocast for the wrapped layers."""
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        return None
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def wrap_for_distributed(model, args, local_rank: int, distributed: bool):
    """Apply DDP or FSDP based on flags. Returns the wrapped module (or unwrapped if single-process)."""
    if not distributed:
        return model.cuda(local_rank) if torch.cuda.is_available() else model

    if args.fsdp:
        layer_cls = find_layer_class(model, args.fsdp_transformer_layer)
        auto_wrap = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={layer_cls},
        )
        return FSDP(
            model,
            auto_wrap_policy=auto_wrap,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=build_mp_policy(args),
            cpu_offload=CPUOffload(offload_params=args.fsdp_cpu_offload),
            device_id=local_rank,
        )

    return DDP(
        model.cuda(local_rank),
        device_ids=[local_rank],
        find_unused_parameters=args.ddp_find_unused_parameters,
    )


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def prepare_dataset(args, tokenizer):
    """Load + tokenize. labels = input_ids for causal LM. Same as the Trainer-based version."""
    raw = (
        load_dataset(args.dataset_name, args.dataset_config)
        if args.dataset_config else load_dataset(args.dataset_name)
    )

    def tokenize_fn(examples):
        encodings = tokenizer(
            examples[args.text_field],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        encodings["labels"] = [ids[:] for ids in encodings["input_ids"]]
        return encodings

    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw[args.train_split].column_names,
    )
    eval_ds = tokenized[args.eval_split] if args.eval_split in tokenized else None
    return tokenized[args.train_split], eval_ds


def make_distributed_dataloader(dataset, batch_size, shuffle, num_workers, world_size, rank, seed):
    """DataLoader with DistributedSampler (or plain sampler if single-process)."""

    def collate(batch):
        return {
            key: torch.tensor([sample[key] for sample in batch], dtype=torch.long)
            for key in batch[0]
        }

    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed,
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
            collate_fn=collate, pin_memory=True,
        )
        return loader, sampler

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate, pin_memory=True,
    )
    return loader, None


# ------------------------------------------------------------------
# Loss + LR schedule
# ------------------------------------------------------------------

def causal_lm_loss(logits, labels):
    """Standard shifted cross-entropy (same as HF's internal causal LM loss)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def linear_schedule_lr(step, warmup_steps, total_steps):
    """Linear warmup to 1.0, then linear decay to 0.0."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-reduce a scalar to compute its mean across ranks (for logging only)."""
    if world_size <= 1 or not dist.is_initialized():
        return value
    out = value.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out / world_size


# ------------------------------------------------------------------
# Eval + checkpointing
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, args, world_size: int):
    """Eval loop. Returns mean loss / perplexity averaged across ranks."""
    model.eval()
    autocast_kwargs = _autocast_kwargs(args, for_fsdp=isinstance(model, FSDP))
    total = torch.zeros(2, device=device)  # (sum_loss, count)

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.cuda.amp.autocast(**autocast_kwargs):
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = causal_lm_loss(out.logits, batch["labels"])
        total[0] += loss.detach()
        total[1] += 1

    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)

    avg = (total[0] / total[1].clamp(min=1)).item()
    model.train()
    return {"eval_loss": avg, "eval_ppl": math.exp(avg) if avg < 30 else float("inf")}


def _autocast_kwargs(args, for_fsdp: bool):
    """For DDP we use autocast directly. For FSDP, MixedPrecision policy already handles dtypes."""
    if for_fsdp:
        return dict(enabled=False)
    if args.bf16:
        return dict(enabled=torch.cuda.is_available(), dtype=torch.bfloat16)
    if args.fp16:
        return dict(enabled=torch.cuda.is_available(), dtype=torch.float16)
    return dict(enabled=False)


def save_model_distributed(model, tokenizer, output_dir: str, args, rank: int, distributed: bool):
    """DDP unwraps `model.module`; FSDP gathers a full state-dict on rank 0."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if isinstance(model, FSDP):
        # Gather full weights to rank 0; offload to CPU so we don't OOM.
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
        if rank == 0:
            # Reconstruct an HF model on CPU and save_pretrained for a clean checkpoint.
            config = AutoConfig.from_pretrained(args.model_name, token=args.hf_token)
            fresh = AutoModelForCausalLM.from_config(config)
            fresh.load_state_dict(cpu_state)
            fresh.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        if distributed:
            dist.barrier()
        return

    # DDP / single-process: unwrap and save_pretrained from rank 0.
    if rank == 0:
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    if distributed:
        dist.barrier()


def save_checkpoint(model, tokenizer, args, step: int, rank: int, distributed: bool):
    """Save under output_dir/checkpoint-{step}/ with save_total_limit pruning."""
    ckpt_dir = str(Path(args.output_dir) / f"checkpoint-{step}")
    save_model_distributed(model, tokenizer, ckpt_dir, args, rank, distributed)
    if rank == 0:
        prune_old_checkpoints(args.output_dir, args.save_total_limit)
    if distributed:
        dist.barrier()


def prune_old_checkpoints(output_dir: str, save_total_limit: int):
    """Match HF Trainer's save_total_limit by deleting oldest checkpoints."""
    if save_total_limit <= 0:
        return
    ckpts = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    while len(ckpts) > save_total_limit:
        oldest = ckpts.pop(0)
        shutil.rmtree(oldest, ignore_errors=True)


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train_loop(model, tokenizer, train_loader, train_sampler, eval_loader,
               optimizer, args, device, rank, world_size, distributed):
    """Manual loop: AMP + grad accumulation + LR schedule + cross-rank logging + ckpt."""
    is_fsdp = isinstance(model, FSDP)
    use_amp_scaler = args.fp16 and not args.bf16 and not is_fsdp and torch.cuda.is_available()
    # GradScaler only for DDP+fp16. FSDP uses MixedPrecision; bf16 doesn't need a scaler.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp_scaler)

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = max(1, int(steps_per_epoch * args.num_train_epochs))
    autocast_kwargs = _autocast_kwargs(args, for_fsdp=is_fsdp)

    if is_main_process(rank):
        logger.info(
            "Training: %d optimizer steps over %.2f epochs (grad_accum=%d, steps/epoch=%d, world=%d).",
            total_steps, args.num_train_epochs,
            args.gradient_accumulation_steps, steps_per_epoch, world_size,
        )

    model.train()
    global_step = 0
    micro_step = 0
    running_loss = torch.zeros(1, device=device)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(math.ceil(args.num_train_epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # Required for proper DistributedSampler shuffling.

        for batch in train_loader:
            if global_step >= total_steps:
                break

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.cuda.amp.autocast(**autocast_kwargs):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = causal_lm_loss(out.logits, batch["labels"])
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward() if use_amp_scaler else loss.backward()
            running_loss += loss.detach()
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps != 0:
                continue

            # ---- Gradient clipping ----
            if args.max_grad_norm > 0:
                if use_amp_scaler:
                    scaler.unscale_(optimizer)
                if is_fsdp:
                    # FSDP needs its own clip helper because params are sharded.
                    model.clip_grad_norm_(args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )

            # ---- LR schedule ----
            lr_scale = linear_schedule_lr(global_step, args.warmup_steps, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = args.learning_rate * lr_scale

            if use_amp_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # ---- Logging (all-reduce so rank 0 sees the world-mean loss) ----
            if global_step % args.logging_steps == 0:
                avg = running_loss / args.logging_steps
                avg = reduce_mean(avg, world_size)
                running_loss.zero_()
                if is_main_process(rank):
                    logger.info(
                        "epoch=%d step=%d loss=%.4f lr=%.2e",
                        epoch, global_step, avg.item(), args.learning_rate * lr_scale,
                    )

            # ---- Eval ----
            if (
                eval_loader is not None
                and args.evaluation_strategy == "steps"
                and global_step % args.eval_steps == 0
            ):
                metrics = evaluate(model, eval_loader, device, args, world_size)
                if is_main_process(rank):
                    logger.info("eval @ step=%d: %s", global_step, metrics)

            # ---- Checkpoint ----
            if global_step % args.save_steps == 0:
                save_checkpoint(model, tokenizer, args, global_step, rank, distributed)

        if global_step >= total_steps:
            break

        if eval_loader is not None and args.evaluation_strategy == "epoch":
            metrics = evaluate(model, eval_loader, device, args, world_size)
            if is_main_process(rank):
                logger.info("eval @ epoch=%d: %s", epoch, metrics)

    return global_step


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, distributed = setup_distributed()
    configure_logging(rank)
    torch.manual_seed(args.seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process(rank):
        logger.info("World size: %d | distributed: %s | strategy: %s",
                    world_size, distributed, "FSDP" if args.fsdp else "DDP" if distributed else "single")

    try:
        # ---- Load tokenizer + base model on CPU (FSDP/DDP wrap moves to GPU) ----
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model_name, token=args.hf_token)

        # ---- Wrap (DDP / FSDP / single-process) ----
        model = wrap_for_distributed(model, args, local_rank, distributed)

        # ---- Data ----
        train_ds, eval_ds = prepare_dataset(args, tokenizer)
        train_loader, train_sampler = make_distributed_dataloader(
            train_ds, args.per_device_train_batch_size, shuffle=True,
            num_workers=args.num_workers, world_size=world_size, rank=rank, seed=args.seed,
        )
        eval_loader, _ = (
            make_distributed_dataloader(
                eval_ds, args.per_device_eval_batch_size, shuffle=False,
                num_workers=args.num_workers, world_size=world_size, rank=rank, seed=args.seed,
            ) if eval_ds is not None else (None, None)
        )

        # ---- Optimizer ----
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        if is_main_process(rank):
            logger.info("Starting training (%s strategy).", "FSDP" if args.fsdp else "DDP" if distributed else "single")

        final_step = train_loop(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            train_sampler=train_sampler,
            eval_loader=eval_loader,
            optimizer=optimizer,
            args=args,
            device=device,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
        )

        if eval_loader is not None and args.evaluation_strategy != "no":
            metrics = evaluate(model, eval_loader, device, args, world_size)
            if is_main_process(rank):
                logger.info("Final eval: %s", metrics)

        save_model_distributed(model, tokenizer, args.output_dir, args, rank, distributed)

        if is_main_process(rank):
            logger.info("Done. Trained for %d optimizer steps.", final_step)

    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
