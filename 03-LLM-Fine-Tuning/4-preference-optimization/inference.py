"""Generate text from a DPO/ORPO/KTO-tuned model.

After preference optimization, the model is just an aligned causal LM —
standard text generation works. If the checkpoint contains LoRA adapters
(default for `train.py --use-lora`), they're loaded automatically when
PEFT detects an `adapter_config.json` next to the weights.
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
        description="Generate text from a preference-tuned model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, default="./preference_finetuned")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Required when loading bare LoRA adapters without a merged base.")
    parser.add_argument("--prompt", type=str, default="Explain reinforcement learning in plain English.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    """Load tokenizer + model. If a base model is given, attach LoRA from --model-path on top."""
    if args.base_model:
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map=args.device
        )
        model = PeftModel.from_pretrained(base, args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map=args.device
        )
    return model, tokenizer


def main() -> None:
    """Tokenize prompt -> generate -> decode."""
    args = parse_args()
    model, tokenizer = load_model(args)

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
