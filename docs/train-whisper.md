# train-whisper.py

Fine-tunes a Whisper model on Hebrew speech data with optional QLoRA quantization.

## Usage

```bash
uv run python train-whisper.py [args]
```

## Arguments

### Data

| Argument | Default | Description |
|---|---|---|
| `--use_preprocessed PATH [PATH ...]` | — | Load preprocessed dataset(s) from disk or HF Hub. Mutually exclusive with `--train_datasets`. Supports multiple paths — datasets are interleaved. |
| `--use_preprocessed_probs FLOAT [...]` | — | Sampling probabilities when interleaving multiple preprocessed datasets (must match number of datasets). |
| `--train_datasets DATASET[:SPLIT] [...]` | — | Raw dataset(s) to train on. Use `dataset_name:split` format. Cannot be used with `--use_preprocessed`. |
| `--eval_datasets DATASET[:SPLIT] [...]` | — | Raw dataset(s) to evaluate on. Required when using `--train_datasets`. |
| `--save_processed PATH` | — | Preprocess raw datasets and save to disk instead of training. |
| `--max_eval_set_size INT` | — | Cap the eval set to this many randomly sampled examples. Useful for fast eval during training. |
| `--target_language` | `hebrew` | Language to transcribe. |

### Model

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `openai/whisper-large-v2` | Base model to fine-tune (HF model ID or local path). |
| `--output_model_name` | *(required)* | Name for the output model directory and HF repo. |
| `--use_qlora` | off | Enable QLoRA (8-bit quantization + LoRA). Greatly reduces VRAM. Recommended for large-v3 on 24GB. |
| `--attn_implementation` | — | Set to `sdpa` to use scaled dot-product attention (faster on supported hardware). |
| `--mixed_precision` | — | `bf16`, `fp16`, or `tf32`. `bf16` recommended on Ampere+ GPUs (RTX 3090/4090). |

### Training

| Argument | Default | Description |
|---|---|---|
| `--num_train_epochs` | `10` | Number of full passes over the training data. |
| `--max_steps` | `-1` | Train for exactly this many steps, overriding `--num_train_epochs`. Useful for quick tests. |
| `--learning_rate` | `1e-5` | Peak learning rate. |
| `--warmup_steps` | `500` | Steps over which LR ramps from 0 to `--learning_rate`. |
| `--warmup_ratio` | `0.1` | Alternative warmup specification as fraction of total steps. Overridden by `--warmup_steps`. |
| `--lr_scheduler_type` | `constant_with_warmup` | LR schedule after warmup. `constant_with_warmup` holds LR flat after warmup. |
| `--weight_decay` | `0.05` | L2 regularization on model weights. |
| `--per_device_train_batch_size` | `16` | Batch size per GPU for training. Reduce if OOM. |
| `--per_device_eval_batch_size` | `16` | Batch size per GPU for evaluation. |
| `--gradient_accumulation_steps` | `2` | Accumulate gradients over N steps before updating weights. Effective batch = `batch_size × accumulation_steps`. |
| `--ds_processor_proc_num` | `1` | Number of parallel workers for dataset preprocessing. |
| `--include_timestamps_prob` | `0.5` | Probability of including timestamps in a training sample. |
| `--include_prev_text_prob` | `0.5` | Probability of conditioning on previous transcript text. |
| `--inject_synthetic_timestamps` | off | When timestamps are requested but not in the data, inject synthetic start+end tokens. |
| `--audio_shift_augmentation` | off | When injecting synthetic timestamps, randomize audio shift as augmentation. |

### Checkpointing & Logging

| Argument | Default | Description |
|---|---|---|
| `--logging_steps` | `500` | Log training metrics every N steps. |
| `--eval_steps` | *(logging_steps)* | Run evaluation every N steps. |
| `--save_steps` | `500` | Save a checkpoint every N steps. |
| `--max_checkpoints_to_keep` | — | Delete older checkpoints, keeping only the N most recent. |
| `--save_only_model` | off | Save only model weights, not optimizer state (smaller checkpoints). |
| `--resume_from_checkpoint` | off | Resume from the latest checkpoint in `--output_model_name`. |
| `--resume_from_checkpoint_path PATH` | — | Resume from a specific checkpoint directory. |
| `--ignore_data_skip` | off | When resuming, don't skip already-seen training examples. |
| `--predict_wer` | off | Compute WER during eval (requires running generation). Without this, only loss is computed — faster eval. |
| `--run_name` | — | Run name for experiment tracking (TensorBoard / W&B). If not set, tracking is disabled. |

### Publishing

| Argument | Default | Description |
|---|---|---|
| `--hf_org_name` | `ivrit-ai` | HuggingFace organization to push the model to. |
| `--skip_push_to_hub` | off | Don't push the model to HF Hub. Required if not logged in. |

---

## Reading the Training Logs

### Training log line (every `--logging_steps` steps)

```
{'loss': 0.4323, 'grad_norm': 0.097, 'learning_rate': 0.0001, 'epoch': 0.38}
```

| Field | What it means |
|---|---|
| `loss` | Average cross-entropy loss over the last N steps. Lower is better. Should trend downward over training. |
| `grad_norm` | Magnitude of gradients. Healthy range is roughly 0.05–0.3. A sudden spike (e.g. 10+) means instability; collapse to 0 means learning stopped. |
| `learning_rate` | Current LR. Ramps up during warmup, then stays flat (with `constant_with_warmup`). |
| `epoch` | Fraction of training data seen. `0.38` = 38% through epoch 1. Goes up to `num_train_epochs`. |

The number of log lines between evals equals `eval_steps / logging_steps`. With `--eval_steps 500 --logging_steps 50` you get exactly 10 log lines per eval cycle.

### Eval log line (every `--eval_steps` steps)

```
{'eval_loss': 0.421, 'eval_wer_ortho': 0.340, 'eval_wer': 0.225, 'eval_runtime': 652, ...}
```

| Field | What it means |
|---|---|
| `eval_loss` | Loss on the held-out eval set. Should track training loss. If eval_loss rises while training loss falls, the model is overfitting. |
| `eval_wer` | **Main metric.** Word Error Rate after Hebrew text normalization (strips niqqud, punctuation, RTL chars). `0.225` = 22.5% of words wrong. Lower is better. |
| `eval_wer_ortho` | WER without normalization (raw text). Always higher than `eval_wer`. Useful for spotting punctuation/casing issues. |
| `eval_runtime` | Seconds spent running eval. Only present when `--predict_wer` is set (generation is expensive). |

### Typical training progression

A healthy run looks like:
1. **Warmup phase**: loss drops sharply (e.g. 1.0 → 0.5), LR ramps up
2. **Main training**: loss slowly decreases (e.g. 0.45 → 0.35), grad_norm stable
3. **Eval WER**: improves each eval cycle, eventually plateaus

---

## Example Commands

### Quick test run (10 steps)
```bash
uv run python train-whisper.py \
    --output_model_name my-whisper-test \
    --model_name openai/whisper-small \
    --use_preprocessed /path/to/preprocessed \
    --target_language hebrew \
    --skip_push_to_hub \
    --max_steps 10 --eval_steps 5 --logging_steps 5 --save_steps 10 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_eval_set_size 50 \
    --use_qlora
```

### Full training run (whisper-large-v3, RTX 4090)
```bash
uv run python train-whisper.py \
    --output_model_name my-whisper-he \
    --model_name ivrit-ai/whisper-large-v3 \
    --use_preprocessed /workspace/preprocessed/my-dataset \
    --target_language hebrew \
    --skip_push_to_hub \
    --use_qlora \
    --mixed_precision bf16 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 3 \
    --max_eval_set_size 200 \
    --max_checkpoints_to_keep 2 \
    --logging_steps 50 --eval_steps 500 --save_steps 500 \
    --predict_wer \
    --run_name my-run-v1
```
