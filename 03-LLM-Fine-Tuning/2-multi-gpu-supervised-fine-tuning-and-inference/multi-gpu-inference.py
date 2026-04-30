"""Multi-GPU inference for a causal LM via HF + `device_map="auto"`.

Inference is simpler than training — no distributed setup, no gradients,
just forward passes. Loading the model with `device_map="auto"` lets
`accelerate` shard layers across whatever GPUs are visible.

Run with:
    python multi-gpu-inference.py --prompt "Hello, how are you?"

The bottom of this file documents three additional strategies (data
parallelism, tensor parallelism via vLLM, pipeline parallelism) and the
DeepSpeed-vs-vLLM decision matrix. They're kept as comments because each
requires different optional dependencies and a different launch model;
see `multi-gpu-inference-vllm.py` for a runnable vLLM example.
"""

import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the model, prompt, and generation parameters."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU inference with HF + device_map='auto'.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-13b-hf")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Prompt to feed the model.",
    )
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float16",
        choices=list(DTYPE_MAP.keys()),
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="HF device_map strategy. 'auto' shards across visible GPUs.",
    )

    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    """Load the tokenizer and a sharded model via `device_map='auto'`."""
    logger.info("Loading tokenizer + model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device_map,
        torch_dtype=DTYPE_MAP[args.torch_dtype],
        low_cpu_mem_usage=True,
        token=args.hf_token,
    )
    return model, tokenizer


def main() -> None:
    """Tokenize a prompt, run multi-GPU generation, print the decoded output."""
    args = parse_args()

    if torch.cuda.is_available():
        logger.info("Visible CUDA devices: %d", torch.cuda.device_count())

    model, tokenizer = load_model_and_tokenizer(args)

    # Inputs go to "cuda"; the framework routes them to whichever GPU holds
    # the embedding layer under `device_map="auto"`.
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")

    gen_kwargs = dict(
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens
    else:
        gen_kwargs["max_length"] = args.max_length

    outputs = model.generate(**inputs, **gen_kwargs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()


# =============================================================================
# Reference: Common Inference Strategies
# =============================================================================
#
# 1. Data Parallelism (model fits on 1 GPU)
#    Assign a model copy to each GPU manually, distribute requests across GPUs:
#    ```python
#    device = torch.device(f"cuda:{gpu_id}")  # gpu_id = 0, 1, 2, 3...
#    model.to(device)
#    ```
#    In production, use vLLM or TGI:
#    ```python
#    from vllm import LLM
#    llm = LLM(
#        model="meta-llama/Llama-2-7b-hf",
#        tensor_parallel_size=4,  # Use 4 GPUs
#    )
#    outputs = llm.generate(["prompt1", "prompt2", ...])
#    ```
#    See `multi-gpu-inference-vllm.py` for a runnable example.
#
# 2. Tensor Parallelism (model too large for 1 GPU)
#    Split model layers across GPUs:
#    ```python
#    model = AutoModelForCausalLM.from_pretrained(
#        "meta-llama/Llama-2-70b-hf",
#        device_map="auto",            # Automatically distributes layers
#        torch_dtype=torch.float16,
#    )
#    ```
#    This is what the script above already does.
#
# 3. Pipeline Parallelism
#    Different stages of the model on different GPUs, requests stream through:
#    ```python
#    from transformers import pipeline
#    pipe = pipeline(
#        "text-generation",
#        model="model_name",
#        device_map="auto",
#    )
#    outputs = pipe("Your prompt")
#    ```
#
# =============================================================================
# DeepSpeed
# =============================================================================
# Microsoft's optimization library — makes multi-GPU training/inference faster
# and more memory-efficient than standard PyTorch. To use:
# * Create a deepspeed config (batch size, gradient_accumulation_steps, fp16,
#   etc.).
# * Pass it via `TrainingArguments(deepspeed=...)`.
# * Launch with `deepspeed --num_gpus=4 train.py`.
#
# TRAINING:
# * Use DeepSpeed when the model doesn't fit even with FSDP, when training
#   very large models (70B+), when you need CPU/NVMe offloading, or when you
#   want maximum memory efficiency.
# * Stick with PyTorch FSDP for smaller models (<13B), simpler setup, or
#   models without DeepSpeed kernel support.
#
# INFERENCE:
# * Use DeepSpeed for maximum throughput, latency-critical applications, or
#   very large models needing tensor parallelism.
# * Use vLLM/TGI for easier setup with better defaults, dynamic batching, or
#   production serving (vLLM is the popular default).
