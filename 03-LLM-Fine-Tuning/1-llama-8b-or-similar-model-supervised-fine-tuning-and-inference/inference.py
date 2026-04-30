"""Run text generation against a fine-tuned Llama 3 8B checkpoint.

Mirrors the article's evaluation step: load a fine-tuned model via the
Hugging Face `text-generation` pipeline and generate completions for a prompt.
"""

import argparse
import logging

from transformers import pipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the model path and generation parameters."""
    parser = argparse.ArgumentParser(
        description="Generate text from a fine-tuned Llama checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, default="./llama_finetuned")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantum computing in simple terms.",
    )
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--device", type=int, default=0, help="CUDA device id, or -1 for CPU.")
    return parser.parse_args()


def main() -> None:
    """Build a text-generation pipeline from the saved model and print completions."""
    args = parse_args()

    logger.info("Loading pipeline from %s", args.model_path)
    nlp = pipeline("text-generation", model=args.model_path, device=args.device)

    results = nlp(
        args.prompt,
        max_length=args.max_length,
        num_return_sequences=args.num_return_sequences,
    )
    for i, result in enumerate(results):
        print(f"--- Completion {i + 1} ---")
        print(result["generated_text"])


if __name__ == "__main__":
    main()
