"""Multi-GPU batch inference with vLLM (tensor-parallel + paged attention).

vLLM is the de facto production serving stack: it shards a model across GPUs
with `tensor_parallel_size`, batches requests dynamically, and uses paged
attention to keep KV-cache memory under control. Throughput is typically
several times higher than vanilla `model.generate` for batched workloads.

Run:
    python multi-gpu-inference-vllm.py --tensor-parallel-size 4 \\
        --prompts "Hello" "Tell me a joke" "Explain transformers"

For a full HTTP server with an OpenAI-compatible API, prefer vLLM's built-in
launcher instead of this script:
    vllm serve meta-llama/Llama-3.1-8b-hf --tensor-parallel-size 4
"""

import argparse
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the model, prompts, parallelism, and sampling."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU batch inference with vLLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8b-hf")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Number of GPUs to shard the model across.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of each GPU's VRAM vLLM may claim for the KV cache.",
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=[
            "Hello, how are you?",
            "Explain quantum computing in one sentence.",
            "Write a haiku about GPUs.",
        ],
        help="One or more prompts to generate completions for.",
    )

    # Sampling
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--n", type=int, default=1, help="Completions per prompt.")

    return parser.parse_args()


def main() -> None:
    """Load the model with tensor parallelism, batch-generate, print results."""
    # Imported lazily so `--help` works even without vLLM installed.
    from vllm import LLM, SamplingParams

    args = parse_args()

    logger.info(
        "Loading vLLM model %s (tensor_parallel_size=%d)",
        args.model_name,
        args.tensor_parallel_size,
    )
    llm_kwargs = dict(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        n=args.n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    logger.info("Generating completions for %d prompt(s)", len(args.prompts))
    outputs = llm.generate(args.prompts, sampling_params)

    for output in outputs:
        print("=" * 60)
        print(f"Prompt: {output.prompt}")
        for i, completion in enumerate(output.outputs):
            print(f"--- Completion {i + 1} ---")
            print(completion.text)


if __name__ == "__main__":
    main()
