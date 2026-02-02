# FINE-TUNING ON 4 GPUs (HF & PyTorch)
# * __Multiple NVIDIA GPUs__ (4x A100, V100, or RTX 4090s)
# * __Connected via NVLink__ (NVidia proprietory GPU-GPU interconnect) or __PCIe (standard GPU-CPU bus)__  
#   (interconnect technologies that allow GPUs to communicate)
# * Use __Distributed Data Parallel (DDP)__ if model fits __on 1 GPU__ - replicate across all GPUs:
#   * Each GPU gets _a copy of model_, _data split_ across GPUs as batches
#   * _Gradients are synchronized_ after backward pass
#   * Handled automatically by TrainingArguments when multiple GPUs detected
# * Use __FSDP (Fully Sharded Data Parallel)__ if model __too large for 1 GPU__ - shard across all GPUs
# * __PyTorch handles synchronization__, you just launch with torchrun
# 
# To tun the below code:  
# `torchrun --nproc_per_node=4 train.py`  # For 4 GPUs on one machine

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare dataset
dataset = load_dataset("your_dataset")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments for multi-GPU
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,  # Mixed precision training
    ddp_find_unused_parameters=False,
    # Multi-GPU settings are auto-detected )

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],)
trainer.train()

# For larger models (FSDP - Fully Sharded Data Parallel):
training_args = TrainingArguments(
       fsdp="full_shard auto_wrap",  # Shard model across GPUs
       fsdp_config={
           "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"]})