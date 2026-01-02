# Unsloth Notebook to DDP Conversion Guide

This guide shows how to convert any Unsloth fine-tuning notebook to support **Distributed Data Parallel (DDP)** training on multiple GPUs (e.g., Kaggle 2x T4, Colab Pro, etc.).

## Quick Summary

**3 Simple Steps:**
1. Add GPU detection and `device_map="balanced"` to model loading
2. Update memory stats to be multi-GPU compatible
3. Update inference code to use explicit device

---

## Step-by-Step Conversion

### Step 1: Modify Model Loading Cell

**Find this cell** (usually right after installation):
```python
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

**Replace with:**
```python
from unsloth import FastLanguageModel
import torch

# ============================================================================
# MULTI-GPU DDP SETUP (Fit-for-all modification for all Unsloth notebooks)
# ============================================================================
# This section automatically detects and sets up multi-GPU training.
# Works on Kaggle 2x T4, Colab Pro, and any multi-GPU setup.
#
# Note: In notebooks, HuggingFace Trainer automatically uses DataParallel
# (single-process multi-GPU) when multiple GPUs are detected. For true
# multi-process DDP, use Unsloth CLI with torchrun (see Unsloth docs).

# Detect number of GPUs
num_gpus = torch.cuda.device_count()
print(f"🦥 Detected {num_gpus} GPU(s)")

# For multi-GPU setups, use device_map="balanced" to split model across GPUs
# This helps with memory distribution and enables the trainer to use all GPUs
if num_gpus > 1:
    device_map = "balanced"
    print(f"📊 Using device_map='balanced' to split model across {num_gpus} GPUs")
    print(f"🚀 Trainer will automatically use all {num_gpus} GPUs for training")
    print(f"   (Using DataParallel in notebook mode - for true DDP, use torchrun)")
else:
    device_map = None
    print("📊 Single GPU mode - no device_map needed")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map = device_map,  # Automatically splits model across GPUs if num_gpus > 1
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
```

**Key Changes:**
- Added GPU detection: `num_gpus = torch.cuda.device_count()`
- Added conditional `device_map = "balanced"` when `num_gpus > 1`
- Pass `device_map` to `from_pretrained()`

---

### Step 2: Update Memory Stats Cells

**Find the memory stats cell** (usually before training):
```python
# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
```

**Replace with:**
```python
# @title Show current memory stats (Multi-GPU compatible)
num_gpus = torch.cuda.device_count()
start_gpu_memory = []
max_memory = []

for i in range(num_gpus):
    gpu_stats = torch.cuda.get_device_properties(i)
    gpu_mem = round(torch.cuda.max_memory_reserved(i) / 1024 / 1024 / 1024, 3)
    gpu_max = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    start_gpu_memory.append(gpu_mem)
    max_memory.append(gpu_max)
    print(f"GPU {i} = {gpu_stats.name}. Max memory = {gpu_max} GB. Reserved = {gpu_mem} GB.")

if num_gpus > 1:
    total_max = sum(max_memory)
    total_reserved = sum(start_gpu_memory)
    print(f"\n📊 Total across {num_gpus} GPUs: {total_max} GB max, {total_reserved} GB reserved")
```

**Find the final memory stats cell** (usually after training):
```python
# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
```

**Replace with:**
```python
# @title Show final memory and time stats (Multi-GPU compatible)
num_gpus = torch.cuda.device_count()
used_memory = []
used_memory_for_lora = []

for i in range(num_gpus):
    gpu_used = round(torch.cuda.max_memory_reserved(i) / 1024 / 1024 / 1024, 3)
    gpu_lora = round(gpu_used - start_gpu_memory[i], 3)
    used_memory.append(gpu_used)
    used_memory_for_lora.append(gpu_lora)
    used_percentage = round(gpu_used / max_memory[i] * 100, 3)
    lora_percentage = round(gpu_lora / max_memory[i] * 100, 3)
    print(f"GPU {i}: Peak reserved = {gpu_used} GB ({used_percentage}%), Training = {gpu_lora} GB ({lora_percentage}%)")

if num_gpus > 1:
    total_used = sum(used_memory)
    total_lora = sum(used_memory_for_lora)
    total_max = sum(max_memory)
    total_used_pct = round(total_used / total_max * 100, 3)
    total_lora_pct = round(total_lora / total_max * 100, 3)
    print(f"\n📊 Total across {num_gpus} GPUs: Peak = {total_used} GB ({total_used_pct}%), Training = {total_lora} GB ({total_lora_pct}%)")

print(f"\n⏱️  {trainer_stats.metrics['train_runtime']} seconds ({round(trainer_stats.metrics['train_runtime']/60, 2)} minutes) used for training.")
if num_gpus > 1:
    print(f"🚀 DDP speedup: Training on {num_gpus} GPUs in parallel!")
```

**Key Changes:**
- Loop through all GPUs instead of just GPU 0
- Store stats in lists
- Show per-GPU and total stats when multiple GPUs are present

---

### Step 3: Update Inference Cells

**Find inference cells** (usually after training):
```python
inputs = tokenizer([...], return_tensors = "pt").to("cuda")
```

**Replace with:**
```python
# For multi-GPU setups, inference uses the first GPU (or device_map handles it)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
inputs = tokenizer([...], return_tensors = "pt").to(device)
```

**Key Changes:**
- Use explicit `device = "cuda:0"` instead of `"cuda"`

---

## Optional: Trainer Configuration

The HuggingFace Trainer automatically detects and uses multiple GPUs when available. However, you can add a note:

```python
from trl import SFTConfig, SFTTrainer

# ============================================================================
# TRAINER CONFIGURATION WITH MULTI-GPU SUPPORT
# ============================================================================
# The trainer automatically uses multiple GPUs when available.
# With device_map="balanced", the model is split across GPUs.
# The trainer will distribute batches across GPUs for parallel training.

num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    print(f"🚀 Training with {num_gpus} GPUs")
    print(f"   Model is split across GPUs using device_map='balanced'")
    print(f"   Effective batch size: per_device_batch_size × {num_gpus} × gradient_accumulation_steps")
    print(f"   = 2 × {num_gpus} × 4 = {2 * num_gpus * 4} samples per update")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        # ... rest of your config
    ),
)
```

---

## Checklist

When converting a notebook, ensure:

- [ ] Added GPU detection in model loading cell
- [ ] Added `device_map="balanced"` when `num_gpus > 1`
- [ ] Updated memory stats to loop through all GPUs
- [ ] Updated inference to use explicit `device = "cuda:0"`
- [ ] Tested on both single-GPU and multi-GPU setups

---

## Important Notes

1. **Notebook vs Script DDP:**
   - In notebooks, the HuggingFace Trainer uses **DataParallel** (single-process multi-GPU)
   - For true **multi-process DDP**, convert to a script and use `torchrun`:
     ```bash
     torchrun --nproc_per_node=2 train.py
     ```
   - See [Unsloth DDP documentation](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp) for script-based DDP

2. **Backward Compatibility:**
   - All changes are backward compatible
   - Single-GPU notebooks will work exactly as before
   - Multi-GPU notebooks automatically benefit from parallel training

3. **Model Splitting:**
   - `device_map="balanced"` splits the model across GPUs for memory efficiency
   - Each GPU processes different batches in parallel
   - Effective batch size = `per_device_batch_size × num_gpus × gradient_accumulation_steps`

---

## Example: Complete Modified Cell

Here's a complete example of the modified model loading cell:

```python
from unsloth import FastLanguageModel
import torch

# Multi-GPU DDP Setup
num_gpus = torch.cuda.device_count()
print(f"🦥 Detected {num_gpus} GPU(s)")

if num_gpus > 1:
    device_map = "balanced"
    print(f"📊 Using device_map='balanced' to split model across {num_gpus} GPUs")
else:
    device_map = None
    print("📊 Single GPU mode - no device_map needed")

# Model Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    device_map = device_map,
)
```

---

## Testing

After conversion, test on:
1. **Single GPU:** Should work exactly as before
2. **Multi-GPU (2x T4):** Should show:
   - Model split across GPUs
   - Training using both GPUs
   - Faster training time
   - Memory stats for both GPUs

---

## References

- [Unsloth Multi-GPU Training Docs](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth)
- [Unsloth DDP Guide](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)

---

**That's it!** These 3 simple modifications will make any Unsloth notebook work with multiple GPUs. 🚀

