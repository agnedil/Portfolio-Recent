# Multi-GPU Fine-Tuning & Inference Reference

Run Hugging Face / PyTorch workloads across multiple NVIDIA GPUs.
`multi-gpu-fine-tuning.py` is a complete CLI-driven training pipeline that
toggles between DDP and FSDP. `multi-gpu-inference.py` is a shorter reference
file documenting the inference strategies you'd reach for in production
(`device_map="auto"`, vLLM, DeepSpeed).

| File | Purpose |
| --- | --- |
| `multi-gpu-fine-tuning.py` | End-to-end DDP / FSDP fine-tuning with HF `Trainer` + `torchrun` |
| `multi-gpu-fine-tuning-short.py` | Minimal reference scaffold — the same idea in ~50 lines, useful as a teaching/quick-glance version |
| `multi-gpu-inference.py` | CLI-driven HF inference with `device_map="auto"`; bottom of file documents data / tensor / pipeline parallelism + DeepSpeed tradeoffs |
| `multi-gpu-inference-short.py` | Minimal reference scaffold — `device_map="auto"` inference + the same parallelism / DeepSpeed notes in ~90 lines, useful as a teaching/quick-glance version |
| `multi-gpu-inference-vllm.py` | Production-style batch inference with vLLM (tensor parallelism + paged attention) |
| `pprint-multi-gpu-fine-tuning-and-inference.ipynb` | Same content rendered for reading in a notebook |

> **Hardware assumed:** 4× NVIDIA GPUs (A100 / V100 / RTX 4090 class)
> connected via NVLink or PCIe. Most snippets assume the model + a forward
> pass fit into a single GPU; FSDP / tensor parallelism / DeepSpeed are noted
> for the cases where it doesn't.

---

## 1. Environment setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Llama models on Hugging Face are gated; authenticate once:
huggingface-cli login

# Confirm all 4 GPUs are visible
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
```

`vllm` and `deepspeed` are listed as optional in `requirements.txt`; comment
them out if you don't need the production-inference / extreme-scale-training
paths.

---

## 2. Running the scripts

### 2.1 Fine-tuning (`multi-gpu-fine-tuning.py`)

The script uses `torchrun` to spawn one process per GPU. The HF `Trainer`
auto-detects the world size and switches to DDP automatically — no manual
`DistributedDataParallel(model)` wrapping needed.

```bash
# Default: 4 GPUs on a single node, DDP, wikitext-2 demo dataset
torchrun --nproc_per_node=4 multi-gpu-fine-tuning.py

# Different number of GPUs
torchrun --nproc_per_node=8 multi-gpu-fine-tuning.py

# Custom model + dataset
torchrun --nproc_per_node=4 multi-gpu-fine-tuning.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name tatsu-lab/alpaca --dataset-config "" \
    --train-split train --eval-split train \
    --text-field instruction --max-length 1024

# Switch to FSDP for models that don't fit on a single GPU
torchrun --nproc_per_node=4 multi-gpu-fine-tuning.py --fsdp \
    --fsdp-transformer-layer LlamaDecoderLayer

# Multi-node (run on each node, with matching --nnodes/--node_rank)
torchrun --nnodes=2 --node_rank=0 --master_addr=<HOST> --master_port=29500 \
    --nproc_per_node=4 multi-gpu-fine-tuning.py
```

The default config trains on `wikitext-2-raw-v1` so the script is runnable
out of the box; swap in your own dataset / text field for real training runs.
Run `python multi-gpu-fine-tuning.py --help` for the full flag list.

### 2.2 Inference — HF (`multi-gpu-inference.py`)

Inference is simpler than training: `device_map="auto"` lets `accelerate`
shard the model across whatever GPUs are visible. No `torchrun` needed —
just run the script directly.

```bash
# Default: Llama-2-13B sharded across all visible GPUs, single prompt
python multi-gpu-inference.py

# Custom prompt + sampling
python multi-gpu-inference.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --prompt "Explain entropy in two sentences." \
    --max-new-tokens 200 --do-sample --temperature 0.7 --top-p 0.9

# Force bfloat16 on Ampere+
python multi-gpu-inference.py --torch-dtype bfloat16
```

The bottom of the file also documents three reference patterns (manual data
parallelism, pipeline parallelism via `pipeline(...)`, DeepSpeed) as
comments — see the file for code snippets and the DeepSpeed-vs-vLLM
decision matrix.

### 2.3 Inference — vLLM (`multi-gpu-inference-vllm.py`)

For batched / production-grade inference, vLLM's tensor parallelism and
paged attention deliver several times the throughput of vanilla
`model.generate`.

```bash
# Default: Llama-2-7B across 4 GPUs, three demo prompts
python multi-gpu-inference-vllm.py --tensor-parallel-size 4

# Larger model, custom prompts, deterministic sampling
python multi-gpu-inference-vllm.py \
    --model-name meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 8 \
    --temperature 0.0 \
    --max-tokens 256 \
    --prompts "Summarize: ..." "Translate: ..."
```

For an HTTP server with an OpenAI-compatible API, prefer vLLM's built-in
launcher over rolling your own:

```bash
vllm serve meta-llama/Llama-2-7b-hf --tensor-parallel-size 4
```

### 2.4 DeepSpeed launch (optional)

If you opt into DeepSpeed for training, launch with its own driver instead
of `torchrun`:

```bash
deepspeed --num_gpus=4 multi-gpu-fine-tuning.py --deepspeed ds_config.json
```

A minimal `ds_config.json` is not provided here — see the
[DeepSpeed config reference](https://www.deepspeed.ai/docs/config-json/).

---

## 3. What happens at runtime

### 3.1 Common pipeline

Both scripts load a Llama tokenizer + `AutoModelForCausalLM`, then either
train or generate. The multi-GPU behavior is *implicit* — none of the user
code calls `DistributedDataParallel(...)` or shard wrappers manually.
Instead:

- **Training** relies on `torchrun` to fork N processes (one per GPU) and on
  HF `TrainingArguments` to pick the right strategy from the environment
  (DDP by default, FSDP if `fsdp=...` is set).
- **Inference** relies on `device_map="auto"` to pick a layer-to-GPU mapping
  at load time.

PyTorch and `accelerate` handle gradient / activation synchronization
between GPUs over NVLink (or PCIe as fallback).

### 3.2 Per-script behavior

**`multi-gpu-fine-tuning.py`**
1. **Argument parsing** — paths, dataset config, hyperparameters, and the
   DDP/FSDP toggle (`--fsdp`).
2. **Model + tokenizer load** — `AutoModelForCausalLM.from_pretrained` and
   `AutoTokenizer.from_pretrained` (default `meta-llama/Llama-2-7b-hf`); pads
   with EOS if the tokenizer has no pad token. No `device_map` — under
   DDP/FSDP each rank owns its own copy/shard.
3. **Dataset prep** — loads the configured HF dataset (default
   `wikitext`/`wikitext-2-raw-v1`), tokenizes with `truncation=True,
   padding="max_length"`, and sets `labels = input_ids` for causal LM.
4. **Training arguments** — `per_device_train_batch_size=4`,
   `gradient_accumulation_steps=4`, `num_train_epochs=3`, `learning_rate=2e-5`,
   `fp16=True`, `ddp_find_unused_parameters=False`. With 4 GPUs the effective
   batch size is `4 × 4 grad-accum × 4 GPUs = 64`. Pass `--bf16` on Ampere+
   to swap fp16 for bf16.
5. **DDP path (default)** — when launched via `torchrun --nproc_per_node=4`,
   HF Trainer enables DDP automatically: each rank gets a full model copy
   and a different shard of the data; gradients are all-reduced after each
   backward pass.
6. **FSDP path (`--fsdp`)** — for models too large for a single GPU,
   `fsdp="full_shard auto_wrap"` plus
   `fsdp_transformer_layer_cls_to_wrap=[args.fsdp_transformer_layer]`
   (default `LlamaDecoderLayer`) shards parameters/gradients/optimizer state
   across GPUs at decoder-layer granularity.
7. **Train + evaluate + save** — `Trainer.train()` runs the loop, optional
   `Trainer.evaluate()` prints metrics, `Trainer.save_model()` writes from
   rank 0 only (safe under `torchrun`). Output lands in `--output-dir`
   (default `./output`).

**`multi-gpu-inference.py`**
1. **Argument parsing** — model, prompt, sampling params (`temperature`,
   `top_p`, `top_k`, `do_sample`), dtype, and `device_map` strategy.
2. **Sharded model load** — `AutoModelForCausalLM.from_pretrained` with
   `device_map="auto"` (default), the chosen `torch_dtype`, and
   `low_cpu_mem_usage=True`. `accelerate` decides which decoder layers go
   on which GPU based on free memory at load time.
3. **Generate** — tokenize the prompt, move tensors to `cuda`, call
   `model.generate(...)`. Cross-GPU activations move automatically during
   the forward pass. Use `--max-new-tokens` (preferred) or `--max-length`.
4. **Reference patterns (commented)** — bottom of the file documents three
   alternative strategies and a DeepSpeed-vs-vLLM decision matrix; see the
   file directly.

**`multi-gpu-inference-vllm.py`**
1. **Argument parsing** — model, list of prompts, `tensor_parallel_size`,
   dtype, GPU memory utilization, and sampling params (`max_tokens`,
   `temperature`, `top_p`, `top_k`, `n`).
2. **Engine load** — `vllm.LLM(model=..., tensor_parallel_size=N)` shards
   the weights across N GPUs at startup; vLLM allocates a paged KV cache
   sized by `--gpu-memory-utilization`.
3. **Batch generate** — `llm.generate(prompts, sampling_params)` runs all
   prompts together; vLLM batches them dynamically and returns a list of
   outputs, each with one or more completions (`--n`).
4. **Print** — for each prompt, print every completion. For an HTTP server
   instead of a one-shot batch, use `vllm serve <model>` (vLLM's built-in
   OpenAI-compatible API server) rather than this script.

---

## 4. Choosing a strategy

| Situation | Strategy | What to set |
| --- | --- | --- |
| Model fits on 1 GPU, training | DDP (default) | Just `torchrun --nproc_per_node=N` |
| Model too large for 1 GPU, training | FSDP | `fsdp="full_shard auto_wrap"` + layer wrap class |
| Model way too large / want offloading | DeepSpeed ZeRO-3 | `deepspeed --num_gpus=N` + `ds_config.json` |
| Inference, model fits on 1 GPU | Data parallelism | One process per GPU, or vLLM `tensor_parallel_size=1` × N replicas |
| Inference, model too large for 1 GPU | Tensor parallelism | `device_map="auto"` (HF) or vLLM `tensor_parallel_size=N` |
| Production serving | vLLM or TGI | Built-in batching, paged attention, multi-GPU |

---

## 5. Quick troubleshooting

- **`RuntimeError: CUDA out of memory` during DDP training** — drop
  `per_device_train_batch_size`, raise `gradient_accumulation_steps`, or
  switch to FSDP.
- **Some GPUs unused (`nvidia-smi` shows 0% on a card)** — `torchrun` was
  not used, or `CUDA_VISIBLE_DEVICES` is masking GPUs. Confirm with
  `python -c "import torch; print(torch.cuda.device_count())"`.
- **`device_map="auto"` puts most of the model on GPU 0** — pass a
  `max_memory={0: "20GiB", 1: "20GiB", ...}` dict to balance manually.
- **DDP hangs on `Trainer.train()`** — usually `find_unused_parameters`. The
  script sets `ddp_find_unused_parameters=False`; flip it to `True` if your
  model has conditional branches that skip parameters.
