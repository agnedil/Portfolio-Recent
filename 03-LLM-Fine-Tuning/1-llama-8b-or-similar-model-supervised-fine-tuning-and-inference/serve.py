"""FastAPI server exposing a fine-tuned Llama 3 8B checkpoint via /generate.

Mirrors the article's deployment step. Run with:
    uvicorn serve:app --host 0.0.0.0 --port 8000

Configure the model path with the LLAMA_MODEL_PATH environment variable
(default: ./llama_finetuned).
"""

import os

from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "./llama_finetuned")
MAX_LENGTH = int(os.environ.get("LLAMA_MAX_LENGTH", "100"))


app = FastAPI(title="Llama 3 8B Inference API")

# Load model + tokenizer once at startup so the first request isn't slow.
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


@app.post("/generate")
def generate(prompt: str):
    """Tokenize, generate, and return a decoded completion for `prompt`."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=MAX_LENGTH)
    return {"generated_text": tokenizer.decode(outputs[0])}
