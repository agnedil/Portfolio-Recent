# ### INFERENCE ON 4 GPUs  
# * Simpler (different strategies) - __no distributed training / no gradients__! Just _forward passes_
# * Load model w/device_map="auto" and you're done

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    device_map="auto",          # HANDLES MULTI-GPU AUTOMATICALLY
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")

# Just run it - framework handles GPU distribution
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))


# ## Common Inference Strategies
# 1. __Data Parallelism__ (_Model fits on 1 GPU_)  
# Assign __model copy to each GPU__ manually, distribute requests across GPUs  
# ```python
# ...
# device = torch.device(f"cuda:{gpu_id}")  # gpu_id = 0, 1, 2, 3...
# model.to(device)
# ...
# ```
#     In production use vLLM, TGI
# ```python
# # vLLM automatically handles multi-GPU data parallelism
# from vllm import LLM
# llm = LLM(
#     model="meta-llama/Llama-2-7b-hf",
#     tensor_parallel_size=4  # Use 4 GPUs, )
# outputs = llm.generate(["prompt1", "prompt2", ...])
# ```
# 
# 2. __Tensor Parallelism__ (_Model too large for 1 GPU_)  
# __Split model layers across GPUs__
# ```python
# from transformers import AutoModelForCausalLM
# # Automatically shard model across all GPUs
# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-2-70b-hf",
#     device_map="auto",  # Automatically distributes layers
#     torch_dtype=torch.float16,)
# ```
# 
# 3. __Pipeline Parallelism__  
# __Different stages of model on different GPUs__, process multiple requests in pipeline:
# ```python
# from transformers import pipeline
# # Pipeline automatically handles device placement
# pipe = pipeline(
#     "text-generation",
#     model="model_name",
#     device_map="auto",)
# outputs = pipe("Your prompt")
# ```

# ### DEEPSPEED
# Microsoft's optimization lib to makes multi-GPU training/inference  
# faster and more memory-efficient than standard PyTorch.
# * Create deepspeed config (batch size, gradient_accumulation_steps, fp16 & other params)
# * In PyTorch, add it to TrainingArguments() and import deepspeed
# * Run it with `deepspeed --num_gpus=4 train.py`
# 
# __TRAINING__  
# ✅ Use DeepSpeed when:
# * Model doesn't fit in GPU memory (even with FSDP)
# * Training very large models (70B+ parameters)
# * Need CPU/NVMe offloading
# * Want maximum memory efficiency  
# 
# ❌ Stick with PyTorch FSDP when:
# * Smaller models (<13B parameters)
# * Want simpler setup
# Using models without DeepSpeed kernel support
# 
# __INFERENCE__  
# ✅ Use DeepSpeed when:
# * Need maximum throughput
# * Latency-critical applications
# * Large models requiring tensor parallelism
# 
# ❌ Use alternatives (vLLM, TGI) when:
# * Want easier setup with better defaults
# * Need dynamic batching
# * Production serving (vLLM is more popular)