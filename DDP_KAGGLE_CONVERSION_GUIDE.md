# DDP Training Guide for Unsloth on Kaggle 2x T4 GPUs

This guide explains how to run ** Distributed Data Parallel (DDP)** training with Unsloth notebooks on Kaggle's 2x T4 GPU setup.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step-by-Step Setup](#step-by-step-setup)
3. [Code Modifications](#code-modifications)
4. [Running with torchrun](#running-with-torchrun)
5. [Troubleshooting](#troubleshooting)
6. [Performance Tips](#performance-tips)

---

## Prerequisites

1. **Kaggle Notebook with 2x T4 GPUs enabled**
2. **Unsloth notebook** (any model - Llama, Mistral, etc.)
3. **Basic understanding** of command-line execution

---

## Step-by-Step Setup

### Step 1: Convert Notebook to Python Script

Since DDP requires multi-process execution, you need to convert your notebook to a `.py` script.

**Option A: Manual Conversion**
1. Download your notebook as `.ipynb`
2. Use `jupyter nbconvert` or manually copy cells to a `.py` file
3. Remove markdown cells, keep only code cells

**Option B: Use Kaggle's Script Editor**
1. In Kaggle, go to "Code" → "New Script"
2. Copy code from notebook cells
3. Save as `.py` file

### Step 2: Add DDP Setup Code

Add this at the beginning of your script (after imports):

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize DDP
def setup_ddp():
    """Initialize distributed training"""
    # Get environment variables set by torchrun
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    
    if rank == -1:
        # Not running with DDP
        print("⚠️  Not running with DDP. Use: torchrun --nproc_per_node=2 script.py")
        return None, None, None
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    # Set default device
    torch.cuda.set_device(device)
    
    print(f"✅ DDP initialized: RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")
    print(f"   Process using GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
    
    return rank, local_rank, world_size

# Call setup
rank, local_rank, world_size = setup_ddp()
is_main_process = rank == 0 if rank is not None else True
```

### Step 3: Modify Model Loading

Update your model loading code:

```python
from unsloth import FastLanguageModel

# DDP-aware device mapping
if local_rank is not None:
    # DDP mode: each process uses its own GPU
    device_map = {"": local_rank}  # Put model on this process's GPU
    print(f"📊 DDP mode: Loading model on GPU {local_rank}")
else:
    # Single GPU or DataParallel mode
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        device_map = "auto"  # Split across GPUs
        print(f"📊 Multi-GPU mode: Using device_map='auto'")
    else:
        device_map = None
        print("📊 Single GPU mode")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",  # Your model
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map=device_map,
)
```

### Step 4: Wrap Model with DDP

After adding LoRA adapters, wrap the model:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    # ... other parameters
)

# Wrap with DDP if using distributed training
if local_rank is not None:
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    print(f"✅ Model wrapped with DDP on GPU {local_rank}")
```

### Step 5: Update Trainer Configuration

Modify your trainer args:

```python
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model.module if hasattr(model, 'module') else model,  # Unwrap DDP for trainer
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        # DDP-specific settings
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        # Important: Only main process should save/log
        save_strategy="steps" if is_main_process else "no",
        logging_strategy="steps",
    ),
)
```

### Step 6: Add Cleanup

At the end of your script:

```python
# Cleanup DDP
if rank is not None:
    dist.destroy_process_group()
    print("✅ DDP cleanup complete")
```

---

## Running with torchrun

### In Kaggle Notebook

Add a new code cell at the end:

```python
# Convert notebook to script and run with DDP
import subprocess
import sys

# Save current notebook cells to script
script_content = """
# Paste your notebook code here, or use nbconvert
"""

# Write script
with open("train_ddp.py", "w") as f:
    f.write(script_content)

# Run with torchrun
!torchrun --nproc_per_node=2 --standalone train_ddp.py
```

### In Kaggle Script (Recommended)

1. Create a new **Script** (not Notebook) in Kaggle
2. Paste your converted code
3. In the script settings, add this to "Script Settings" → "Command":

```bash
torchrun --nproc_per_node=2 --standalone train_ddp.py
```

Or run directly in a code cell:

```python
!torchrun --nproc_per_node=2 --standalone /kaggle/working/train_ddp.py
```

---

## Code Modifications Summary

### Required Changes:

1. **Add DDP initialization** (beginning of script)
2. **Modify device_map** for DDP: `{"": local_rank}`
3. **Wrap model with DDP** after LoRA setup
4. **Update trainer args** with DDP settings
5. **Unwrap model** for trainer: `model.module`
6. **Add cleanup** at the end

### Compatibility Fixes (Always Include)

Add this compatibility cell before model loading:

```python
# ============================================================================
# COMPATIBILITY FIXES FOR PEFT AND MULTI-GPU
# ============================================================================
from peft import LoraConfig
import inspect

# Fix 1: PEFT ensure_weight_tying compatibility
sig = inspect.signature(LoraConfig.__init__)
has_ensure_weight_tying = 'ensure_weight_tying' in sig.parameters

if not has_ensure_weight_tying:
    original_init = LoraConfig.__init__
    def patched_init(self, *args, **kwargs):
        kwargs.pop('ensure_weight_tying', None)
        return original_init(self, *args, **kwargs)
    LoraConfig.__init__ = patched_init
    print("✓ Patched LoraConfig for ensure_weight_tying compatibility")

# Fix 2: Multi-GPU attention device placement (if using device_map="auto")
if torch.cuda.device_count() > 1:
    try:
        from xformers.ops.fmha.common import Inputs
        from xformers.ops.fmha import _memory_efficient_attention_forward
        
        original_validate = Inputs.validate_inputs
        def patched_validate_inputs(self):
            if self.attn_bias is not None:
                query_device = self.query.device
                if hasattr(self.attn_bias, 'q_seqinfo'):
                    if hasattr(self.attn_bias.q_seqinfo, 'seqstart'):
                        if self.attn_bias.q_seqinfo.seqstart.device != query_device:
                            self.attn_bias.q_seqinfo.seqstart = \
                                self.attn_bias.q_seqinfo.seqstart.to(query_device)
            return original_validate(self)
        
        Inputs.validate_inputs = patched_validate_inputs
        
        original_forward = _memory_efficient_attention_forward
        def patched_forward(inp, op=None):
            if inp.attn_bias is not None and inp.query is not None:
                query_device = inp.query.device
                if hasattr(inp.attn_bias, 'q_seqinfo'):
                    if hasattr(inp.attn_bias.q_seqinfo, 'seqstart'):
                        if inp.attn_bias.q_seqinfo.seqstart.device != query_device:
                            inp.attn_bias.q_seqinfo.seqstart = \
                                inp.attn_bias.q_seqinfo.seqstart.to(query_device)
            return original_forward(inp, op)
        
        import xformers.ops.fmha
        xformers.ops.fmha._memory_efficient_attention_forward = patched_forward
        print("✓ Patched attention mechanism for multi-GPU compatibility")
    except Exception as e:
        print(f"⚠ Could not patch attention: {e}")
```

---

## Troubleshooting

### Issue 1: "NCCL not found" or "backend not available"

**Solution:**
```python
# Ensure NCCL is available (should be on Kaggle)
print(f"NCCL available: {dist.is_nccl_available()}")
# If False, you may need to use 'gloo' backend (slower)
backend = "nccl" if dist.is_nccl_available() else "gloo"
```

### Issue 2: "Address already in use"

**Solution:**
```python
# Add this before dist.init_process_group()
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'  # Change if port in use
```

### Issue 3: "CUDA out of memory"

**Solutions:**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Use `load_in_4bit=True`
- Reduce `max_seq_length`

### Issue 4: Only one GPU being used

**Check:**
```python
# Verify DDP is active
print(f"RANK: {os.environ.get('RANK')}")
print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")

# Check GPU usage
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
```

### Issue 5: Model not syncing across processes

**Solution:**
```python
# Ensure model is wrapped correctly
if local_rank is not None:
    model = DDP(model, device_ids=[local_rank], 
                find_unused_parameters=False,
                broadcast_buffers=True)
```

---

## Performance Tips

1. **Batch Size**: With 2 GPUs, effective batch = `per_device_batch_size × 2 × gradient_accumulation_steps`
   - Example: `per_device=2, grad_accum=4` → effective batch = 16

2. **Gradient Accumulation**: Use to simulate larger batches without OOM
   - DDP already multiplies by number of GPUs

3. **Mixed Precision**: Unsloth handles this automatically, but you can verify:
   ```python
   print(f"Using dtype: {model.dtype}")
   ```

4. **Monitoring**: Only main process should log/save:
   ```python
   if is_main_process:
       # Logging, saving, etc.
   ```

5. **Data Loading**: Ensure dataset is properly sharded:
   ```python
   # Trainer handles this automatically, but verify:
   print(f"Dataset size: {len(dataset)}")
   print(f"Expected per process: {len(dataset) // world_size}")
   ```

---

## Complete Example Script Structure

```python
#!/usr/bin/env python3
"""
Unsloth DDP Training Script for Kaggle 2x T4
Run with: torchrun --nproc_per_node=2 script.py
"""

# 1. Imports
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

# 2. DDP Setup
def setup_ddp():
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    
    if rank == -1:
        print("⚠️  Not running with DDP")
        return None, None, None
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    print(f"✅ DDP: RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")
    return rank, local_rank, world_size

rank, local_rank, world_size = setup_ddp()
is_main_process = rank == 0 if rank is not None else True

# 3. Compatibility Fixes
# ... (paste compatibility code from above)

# 4. Model Loading
device_map = {"": local_rank} if local_rank is not None else "auto"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map=device_map,
)

# 5. LoRA Setup
model = FastLanguageModel.get_peft_model(model, r=16, ...)

# 6. DDP Wrapping
if local_rank is not None:
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

# 7. Data Preparation
# ... (your data prep code)

# 8. Trainer Setup
trainer = SFTTrainer(
    model=model.module if hasattr(model, 'module') else model,
    # ... (your trainer config)
)

# 9. Training
if is_main_process:
    print("🚀 Starting training...")
trainer_stats = trainer.train()

# 10. Save (only main process)
if is_main_process:
    model.module.save_pretrained("lora_model") if hasattr(model, 'module') else model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")

# 11. Cleanup
if rank is not None:
    dist.destroy_process_group()
```

---

## Quick Reference

### Command to Run:
```bash
torchrun --nproc_per_node=2 --standalone your_script.py
```

### Key Environment Variables:
- `RANK`: Global process rank (0, 1, 2, ...)
- `LOCAL_RANK`: Local GPU index (0, 1)
- `WORLD_SIZE`: Total number of processes (2 for 2 GPUs)

### Expected Speedup:
- **2x T4 with DDP**: ~1.8-1.9x faster than single GPU
- **2x T4 with DataParallel**: ~1.5-1.6x faster than single GPU

---

## Additional Resources

- [Unsloth DDP Documentation](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Kaggle GPU Documentation](https://www.kaggle.com/docs/notebooks)

---

**Last Updated**: 2024
**Compatible with**: Unsloth 2024.8+, PyTorch 2.0+, Kaggle 2x T4
