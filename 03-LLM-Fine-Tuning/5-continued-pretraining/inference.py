"""Generate text from a CPT-adapted model.

The output of CPT is a regular causal LM that knows more about your domain.
Generation works like any HF causal LM. CPT models are *not* instruction-
tuned by default — if you started from a base (non-instruct) model and want
chat behavior, run an SFT pass on top.
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the model path and generation parameters."""
    parser = argparse.ArgumentParser(
        description="Generate text from a CPT-adapted model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, default="./cpt_finetuned")
    parser.add_argument("--prompt", type=str, default="In the field of")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    """Tokenize -> generate -> decode."""
    args = parse_args()

    logger.info("Loading model from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map=args.device
    )

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
