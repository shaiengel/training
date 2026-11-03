# Whisper Fine-Tuning Guide

## Dataset
- Total samples: 109,908
- Split:
  - Train: 90% = ~98,917 samples
  - Validation: 5% = ~5,495 samples
  - Test: 5% = ~5,495 samples

---

## Step 1: Find a Good Learning Rate

Before full training, run a short learning rate search.

> **Note:** The common "10% of dataset" rule is meant to land around 1,000–2,000 samples. With a large dataset like this one, 10% would be ~10,000 samples which is unnecessarily large for LR search. Use a fixed number instead.

1. Use a fixed **2,000 samples** (regardless of total dataset size)
2. Batch size = 16
3. Steps = 2,000 ÷ 16 = 125 steps (one pass through the subset)
4. Either:
   - Run a LR range test: start from `1e-8`, increase exponentially each step, pick the LR just before loss explodes
   - Or try a few fixed candidates: `1e-5`, `5e-5`, `1e-4` and compare which drives loss down most smoothly
5. Typical good LR for Whisper fine-tuning: `1e-5` to `1e-4`

---

## Step 2: Full Fine-Tuning

### Key Training Parameters
```python
batch_size = 16
learning_rate = <value found in step 1>
num_train_epochs = <3-10, depending on results>
evaluation_strategy = "steps"
eval_steps = 1200         # evaluate every ~1,200 steps (every ~20% of an epoch)
save_steps = 1200         # save checkpoint at same frequency
save_total_limit = 2      # only keep the 2 best checkpoints on disk
load_best_model_at_end = True
metric_for_best_model = "wer"   # HuggingFace will look for eval_wer automatically
greater_is_better = False       # lower WER is better
```

### Gradient Accumulation (if GPU memory is limited)
If batch size 16 causes OOM errors, use gradient accumulation instead:
```python
per_device_train_batch_size = 8
gradient_accumulation_steps = 2   # effective batch size = 8 x 2 = 16
```

---

## Step 3: Monitor Training Progress

Track these three metrics together at each checkpoint:

| Metric | What it tells you |
|---|---|
| `train_loss` | Is the model learning at all? |
| `eval_loss` | Is it generalizing or overfitting? |
| `eval_wer` | Real-world transcription quality |

### What to watch for:
- `train_loss ↓` + `eval_loss ↓` → good, keep going
- `train_loss ↓` + `eval_loss ↑` → overfitting, stop here ⚠️

### Checkpoint saving logic:
- At each eval step, WER is computed on the **validation set**
- The checkpoint is kept only if its `eval_wer` is better than the current best
- Worse checkpoints are automatically deleted
- `save_total_limit = 2` ensures at most 2 checkpoints exist on disk at any time

---

## Step 4: Final Evaluation

- After training is fully done, evaluate once on the **test set**
- This is the only time the test set is used
- This gives the honest, unbiased final WER score
- Never use the test set during training or for picking checkpoints

---

## Notes
- `eval_` prefix in HuggingFace always refers to the validation set
- `metric_for_best_model = "wer"` and `metric_for_best_model = "eval_wer"` are equivalent in HuggingFace Trainer
- Checkpoints store: model weights, optimizer state, scheduler state, training state — they can be 1–6 GB each depending on Whisper model size
- To save disk space and only keep model weights (for inference only, not resuming training), save just the model weights separately after training ends