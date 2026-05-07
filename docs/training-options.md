# Training Continuation Options

## Current State
- Model: `my-whisper-he` (LoRA adapter on `ivrit-ai/whisper-large-v3`)
- Best WER achieved: **0.1970** at epoch 2.74
- Total epochs run: 8 (3 + resumed 5 more)
- LR scheduler used: `constant_with_warmup` (flat LR throughout — this is why WER plateaued)
- Final model saved: `my-whisper-he/adapter_model.safetensors` = best checkpoint (epoch 2.74 weights)

## Why WER Stopped Improving
The learning rate stayed constant at `1e-4` for all 8 epochs. With no decay, the optimizer keeps taking large steps and oscillates around a minimum instead of settling into it. A cosine schedule decays the LR smoothly toward 0, allowing the model to converge.

---

## Option A — Fresh start from base model with cosine

```
uv run python train-whisper.py --output_model_name my-whisper-he-v2 --model_name ivrit-ai/whisper-large-v3 --use_preprocessed /workspace/preprocessed/portal-daf-yomi --target_language hebrew --use_qlora --learning_rate 1e-4 --per_device_train_batch_size 8 --num_train_epochs 4 --max_eval_set_size 200 --max_checkpoints_to_keep 2 --mixed_precision bf16 --logging_steps 50 --eval_steps 500 --save_steps 500 --run_name portal-daf-yomi-v2 --predict_wer --skip_push_to_hub --lr_scheduler_type cosine
```

**Reasoning:**
- Starts completely fresh from `ivrit-ai/whisper-large-v3`
- Clean optimizer state, cosine LR from step 0
- Most reliable — no interference from previous training
- Takes more epochs to converge since it starts from scratch
- Start with 4 epochs to validate WER is improving, then extend with `--resume_from_checkpoint --num_train_epochs 12`

**When to choose:** You want the most reliable comparison and don't mind longer training time.

---

## Option B — Warm start from best LoRA weights with cosine

```
uv run python train-whisper.py --output_model_name my-whisper-he-v2 --model_name my-whisper-he --use_preprocessed /workspace/preprocessed/portal-daf-yomi --target_language hebrew --use_qlora --learning_rate 5e-5 --per_device_train_batch_size 8 --num_train_epochs 8 --max_eval_set_size 200 --max_checkpoints_to_keep 2 --mixed_precision bf16 --logging_steps 50 --eval_steps 500 --save_steps 500 --run_name portal-daf-yomi-v2 --predict_wer --skip_push_to_hub --lr_scheduler_type cosine
```

**Reasoning:**
- `--model_name my-whisper-he` loads `adapter_model.safetensors` (epoch 2.74 weights — the best checkpoint)
- No `--resume_from_checkpoint` → fresh optimizer, cosine schedule starts from step 0
- Lower LR (`5e-5`) because the weights are already trained — large steps could damage them
- Converges faster than Option A since it starts from a better point
- Risk: the model may be stuck in a local minimum from previous training

**When to choose:** You want faster convergence and trust that the epoch 2.74 weights are a good starting point.

---

## Key Differences

| | Option A | Option B |
|---|---|---|
| Starting weights | Base model (random LoRA) | Your best trained LoRA (epoch 2.74) |
| Optimizer state | Fresh | Fresh |
| LR scheduler | Cosine from step 0 | Cosine from step 0 |
| Starting LR | `1e-4` | `5e-5` |
| Epochs needed | More (~12) | Fewer (~8) |
| Risk | Longer training | May be stuck in local minimum |

---

## What NOT to Do
- **Do not** use `--resume_from_checkpoint` with a changed LR scheduler — the old optimizer momentum (built under flat LR) conflicts with the new schedule
- **Do not** use `--num_train_epochs` equal to or less than the resumed checkpoint epoch — training exits immediately
- **Do not** reuse `--output_model_name my-whisper-he` — it will overwrite your current best model

---

## Extending a Run After Validation
If after 4 epochs the WER is clearly improving, extend without starting over:

```
uv run python train-whisper.py --output_model_name my-whisper-he-v2 ... --num_train_epochs 12 --resume_from_checkpoint
```

The `--resume_from_checkpoint` here is safe because the optimizer state was built under cosine from the start.
