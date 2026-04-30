# Knowledge Distillation

Train a small **student** model to mimic a large **teacher**. Online logit
distillation: at every batch, both models forward-pass on the same inputs;
the student is updated to match the teacher's softened token-level
probability distribution (KL divergence), optionally combined with the
standard cross-entropy on ground-truth labels.

| File | Purpose |
| --- | --- |
| `train.py` | Distill a student from a teacher with KL + CE loss |
| `inference.py` | Generate text from the distilled student |

---

## 1. What is distillation?

Hinton et al. (2015) showed that a small model can learn faster and
generalize better when its training signal is the *output distribution* of
a larger model rather than just hard labels. Two key knobs:

- **Temperature `T`** — softens both teacher and student logits before the
  KL. Higher T = more information from non-top tokens. Common values: 1–4.
- **`alpha`** — balances the two losses:
  `loss = alpha * KL(student || teacher) + (1 - alpha) * CE(student, labels)`.
  When `alpha = 1.0`, the student learns purely from the teacher.

This script does **online logit distillation** — both models run together
each step. Two practical alternatives exist (and could be added later):

- **Offline / response distillation** — generate teacher outputs once,
  then SFT the student on `(input, teacher_output)` pairs. Cheaper but
  loses the soft-label signal; effectively just SFT under another name.
- **Black-box distillation** — same as offline but the teacher is closed-
  source (e.g. GPT-4); you only get sampled outputs, no logits.

### Vocabulary constraint

Token-level distillation requires the teacher and student to share the
**same tokenizer / vocabulary**. The default pair (Llama-3.1-8B-Instruct
teacher → Llama-3.2-1B-Instruct student) shares the Llama 3 tokenizer.
The script raises an error if the vocab sizes don't match.

If you want to distill across families (e.g. GPT-4 → Llama), you have to
fall back to offline / response distillation or train a vocabulary
projection layer.

### How does this differ from the other fine-tuning methods in this repo?

| Method | Supervision signal | Need a teacher? | Model size after training |
| --- | --- | --- | --- |
| **Distillation** | Teacher logits (soft labels) | Yes | **Smaller** (the student) |
| **SFT** | Ground-truth tokens | No | Same as base |
| **CPT** | Raw text next-token | No | Same as base |
| **DPO / ORPO / KTO** | Human preferences | No | Same as base |
| **GRPO RL** | Programmatic rewards | No | Same as base |

The unique value of distillation is **shrinking** a model while keeping
most of its capability — useful for serving cost, latency, or on-device
deployment.

---

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # Llama models are gated
```

Memory: the teacher and student must both fit on the available GPUs. The
script loads the teacher with `device_map="auto"` so it shards across
visible GPUs automatically; the student trains via the regular HF Trainer.
For the default 8B → 1B pair on a single 40 GB GPU, this works in bf16
without quantization.

---

## 3. Running

### Default demo

```bash
# Llama 3.1 8B teacher -> Llama 3.2 1B student, distilled on Alpaca
python train.py
```

### Custom teacher / student pair

```bash
# Same family is required (shared vocabulary)
python train.py \
    --teacher-model Qwen/Qwen2.5-7B-Instruct \
    --student-model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset-name HuggingFaceH4/ultrachat_200k \
    --max-train-samples 20000
```

### Tuning the distillation knobs

```bash
# Pure teacher signal, higher temperature
python train.py --alpha 1.0 --temperature 4.0

# Heavier weight on ground truth
python train.py --alpha 0.3 --temperature 1.0
```

### Inference

```bash
python inference.py --model-path ./distilled_student \
    --prompt "What is overfitting?"
```

---

## 4. What happens at runtime

1. **Argument parsing** — paths, distillation knobs (`alpha`, `temperature`),
   and the standard training schedule.
2. **Models + tokenizer load** — student in default precision (trainable),
   teacher in bf16 with `device_map="auto"` (frozen, eval-mode). Tokenizer
   is loaded from the student. The script verifies that the two models
   share a vocabulary; otherwise it errors out.
3. **Dataset prep** — `tatsu-lab/alpaca` by default (configurable). Tokenize
   to `--max-length`, set `labels = input_ids` so the student gets a
   standard cross-entropy alongside the distillation term.
4. **DistillationTrainer** — a `Trainer` subclass with a custom
   `compute_loss`:
   - Student forward → CE on labels (`student_outputs.loss`).
   - Teacher forward (`torch.no_grad`) on the same inputs.
   - Both logits are temperature-softened. KL is computed on the
     **shifted** positions (predict token `t+1` from positions up to `t`)
     and scaled by `T²` per Hinton et al.
   - `loss = alpha * KL + (1 - alpha) * CE`.
5. **Train + save** — `trainer.train()` updates the student;
   `trainer.save_model()` writes the student weights to `--output-dir`.
   The teacher is never modified or saved.

---

## 5. Outputs

`--output-dir` (default `./distilled_student`) contains a standalone HF
checkpoint of the student:

```
distilled_student/
├── config.json
├── model.safetensors
├── tokenizer.json / tokenizer_config.json
└── ...
```

Reload with `AutoModelForCausalLM.from_pretrained(...)` — same shape as the
original student, just smarter.
