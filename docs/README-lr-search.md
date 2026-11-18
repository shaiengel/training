# Learning Rate Search on vast.ai

How to find a good learning rate using a small subset of the training data, then move the work to a rented GPU machine. See `find-learning-rate.md` for the reasoning behind the search itself.

Source dataset: `portal-daf-yomi/portal-daf-yomi-whisper-training` (train ~101K, test ~8.88K, ~52 GB).

## 1. Build the subset (once)

`make_subset.py` downloads the source dataset to the HF cache on disk (52 GB, and roughly as much again for the converted Arrow files, so keep ~120 GB free). It then takes a shuffled, seeded sample of 2000 train rows and 300 test rows, saves them to `--out_dir`, and pushes them from there to a private Hub repo. A 2000-row subset is about 1 GB. Nothing large is held in RAM.

Do not use the slice syntax `train[:2000]` for this. It takes the first rows, which are probably ordered by source.

```bash
uv run hf auth login
uv run python make_subset.py --out_dir lr_subset --push_to_hub <your-user>/daf-yomi-lr-subset
```

In VS Code, use the "Make LR Search Subset" launch configuration (`.vscode/launch.json`). Edit its `--push_to_hub` repo id first.

Options: `--train_size` (default 2000), `--test_size` (default 300), `--seed` (default 745), `--out_dir` (required, where the subset is saved). This step needs no GPU, so you can run it on your local machine.

## 2. Set up the vast.ai instance

1. Rent a GPU with at least 60 GB of container disk. The default 32 GB is not enough. For VRAM needs, see "vast.ai Requirements" in the main `README.md`.
2. Clone the repo and install dependencies:
   ```bash
   git clone <repo-url> training && cd training
   uv sync
   ```
3. Point the HF cache at a large disk and log in:
   ```bash
   export HF_HOME=/workspace/.hf_home
   hf auth login
   ```
4. If you see a `libcudnn_ops` error, use the `LD_LIBRARY_PATH` fix from `CLAUDE.md`.

`make_subset.py` must be committed and pushed (or copied with `scp`) so it exists on the instance if you want to run it there.

## 3. Preprocess the subset (once)

Mel-feature extraction is the slow part and does not depend on the learning rate. Do it once with `--save_processed`, and reuse the result for every LR run.

```bash
python train-whisper.py --output_model_name lr-prep --skip_push_to_hub \
  --model_name <whisper-model> \
  --train_datasets <your-user>/daf-yomi-lr-subset:train \
  --eval_datasets  <your-user>/daf-yomi-lr-subset:test \
  --save_processed /workspace/lr_prep --ds_processor_proc_num 4
```

Use the same `--model_name` as in the real run, because the processor depends on it.

## 4. Run one training per candidate LR

```bash
for LR in 1e-5 5e-5 1e-4; do
python train-whisper.py --output_model_name lr-$LR --skip_push_to_hub \
  --model_name <whisper-model> \
  --use_preprocessed /workspace/lr_prep \
  --learning_rate $LR --max_steps 125 \
  --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
  --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 \
  --run_name lr-$LR
done
```

Add `--mixed_precision bf16` on an Ampere or newer GPU.

Why these flags:

- `--max_steps 125` with batch size 16 is one pass over 2000 samples.
- `--gradient_accumulation_steps 1`: the default is 2, which would halve the number of optimizer steps.
- `--warmup_steps 10`: the default is 500, which is longer than the whole search.
- No `--predict_wer`: eval computes loss only, which is much faster than generating text.
- `--run_name` turns on metric reporting, so each run appears as its own curve.
- `--save_steps 100000` avoids writing checkpoints during the search.

## 5. Pick the LR

Compare `train_loss` and `eval_loss` across runs. Pick the highest LR whose loss goes down smoothly. Reject any run where the loss spikes or diverges.

If you use `--use_qlora` (rank 64, alpha 1, RSLoRA), useful LRs are usually higher than for full fine-tuning. Try `1e-4`, `3e-4` and `1e-3` instead of the range above.

## Results: full fine-tuning, whisper-large-v3

Model: `ivrit-ai/whisper-large-v3` (1.55B params). GPU: NVIDIA RTX 6000 Ada / L40S. bf16 training, batch 4 × 4 gradient accumulation (effective batch 16), 125 steps (one epoch over 2000 samples), 10 warmup steps.

Commands used:

```bash
python train-whisper.py --output_model_name lr-prep --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --train_datasets <user>/daf-yomi-lr-subset:train --eval_datasets <user>/daf-yomi-lr-subset:test --save_processed /workspace/lr_subset_processed_large --ds_processor_proc_num 4

uv run python train-whisper.py --output_model_name lr-1e-4 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --use_preprocessed /workspace/lr_subset_processed_large --learning_rate 1e-4 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-1e-4 --mixed_precision bf16

uv run python train-whisper.py --output_model_name lr-5e-5 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --use_preprocessed /workspace/lr_subset_processed_large --learning_rate 5e-5 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-5e-5 --mixed_precision bf16

uv run python train-whisper.py --output_model_name lr-3e-5 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --use_preprocessed /workspace/lr_subset_processed_large --learning_rate 3e-5 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-3e-5 --mixed_precision bf16

uv run python train-whisper.py --output_model_name lr-1e-5 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --use_preprocessed /workspace/lr_subset_processed_large --learning_rate 1e-5 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-1e-5 --mixed_precision bf16

uv run python train-whisper.py --output_model_name lr-5e-6 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --use_preprocessed /workspace/lr_subset_processed_large --learning_rate 5e-6 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-5e-6 --mixed_precision bf16
```

### Eval loss by epoch

| Epoch | 1e-4  | 5e-5  | 3e-5  | 1e-5  | 5e-6  |
|-------|-------|-------|-------|-------|-------|
| 0.2   | 0.562 | 0.468 | 0.438 | 0.421 | 0.430 |
| 0.4   | 0.553 | 0.463 | 0.423 | 0.400 | 0.400 |
| 0.6   | 0.549 | 0.454 | 0.416 | 0.388 | 0.390 |
| 0.8   | 0.545 | 0.449 | 0.408 | 0.382 | 0.384 |
| 1.0   | 0.532 | 0.432 | 0.400 | 0.376 | 0.378 |

### Final train loss

| LR   | 1e-4  | 5e-5  | 3e-5  | 1e-5  | 5e-6  |
|------|-------|-------|-------|-------|-------|
| Loss | 0.553 | 0.462 | 0.433 | 0.417 | 0.427 |

### Analysis

1e-5 wins: lowest eval loss (0.376) and lowest train loss (0.417), still dropping at epoch 1.0, no signs of plateau. 5e-6 is very close (0.378) but converges slightly slower. Higher LRs are worse: 1e-4 and 5e-5 end well above the others (0.532 and 0.432), so the pretrained model benefits from gentle fine-tuning. No run spikes or diverges. These results come from the rerun after the dataset fixes.

Note: batch size here is 4 × 4 = 16 (same effective batch as section 4). If you change the batch size for the real run, the LR may need re-tuning.

## Results: full fine-tuning, whisper-large-v3-turbo

Model: `ivrit-ai/whisper-large-v3-turbo` (809M params). GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition. bf16 training, batch 4 × 4 gradient accumulation (effective batch 16), 125 steps (one epoch over 2000 samples), 10 warmup steps.

Commands used:

```bash
python train-whisper.py --output_model_name lr-prep --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3-turbo --train_datasets <user>/daf-yomi-lr-subset:train --eval_datasets <user>/daf-yomi-lr-subset:test --save_processed /workspace/lr_subset_processed_large_turbo --ds_processor_proc_num 4

uv run python train-whisper.py --output_model_name lr-1e-4 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3-turbo --use_preprocessed /workspace/lr_subset_processed_large_turbo --learning_rate 1e-4 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-1e-4 --mixed_precision bf16

uv run python train-whisper.py --output_model_name lr-5e-5 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3-turbo --use_preprocessed /workspace/lr_subset_processed_large_turbo --learning_rate 5e-5 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-5e-5 --mixed_precision bf16

uv run python train-whisper.py --output_model_name lr-3e-5 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3-turbo --use_preprocessed /workspace/lr_subset_processed_large_turbo --learning_rate 3e-5 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-3e-5 --mixed_precision bf16

uv run python train-whisper.py --output_model_name lr-1e-5 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3-turbo --use_preprocessed /workspace/lr_subset_processed_large_turbo --learning_rate 1e-5 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-1e-5 --mixed_precision bf16

uv run python train-whisper.py --output_model_name lr-5e-6 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3-turbo --use_preprocessed /workspace/lr_subset_processed_large_turbo --learning_rate 5e-6 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-5e-6 --mixed_precision bf16
```

### Eval loss by epoch

| Epoch | 1e-4  | 5e-5  | 3e-5  | 1e-5  | 5e-6  |
|-------|-------|-------|-------|-------|-------|
| 0.2   | 0.752 | 0.598 | 0.541 | 0.511 | 0.522 |
| 0.4   | 0.803 | 0.587 | 0.530 | 0.484 | 0.485 |
| 0.6   | 0.717 | 0.575 | 0.523 | 0.470 | 0.471 |
| 0.8   | 0.720 | 0.569 | 0.509 | 0.463 | 0.460 |
| 1.0   | 0.708 | 0.561 | 0.495 | 0.455 | 0.454 |

### Final train loss

| LR   | 1e-4  | 5e-5  | 3e-5  | 1e-5  | 5e-6  |
|------|-------|-------|-------|-------|-------|
| Loss | 0.731 | 0.583 | 0.535 | 0.506 | 0.518 |

### Analysis

1e-5 wins again: tied with 5e-6 on final eval loss (0.455 vs 0.454) but converges faster (lower train loss 0.506 vs 0.518), still dropping at epoch 1.0. 1e-4 overshoots badly — eval spikes to 0.803 at epoch 0.4 before partially recovering. The turbo model (809M params) shows the same preference for gentle fine-tuning as v3 (1.55B params), with overall higher loss values due to the smaller decoder.

Note: batch size here is 4 × 4 = 16 (same effective batch as section 4). If you change the batch size for the real run, the LR may need re-tuning.

## 6. The real run

Use the full dataset with the chosen LR. Keep the eval set small:

- Pass `--eval_datasets portal-daf-yomi/portal-daf-yomi-whisper-training:test[:500]` and `--max_eval_set_size 500`.
- `--max_eval_set_size` is applied only after the whole eval set has been preprocessed. Slicing the spec as above avoids preprocessing all 8.88K test rows.
- The dataset has no validation split. Use rows from `test` for the small eval set, and keep a separate set of `test` rows that never takes part in training or checkpoint selection for the final score.
- Reset `--warmup_steps`, `--gradient_accumulation_steps` and `--logging_steps` for the longer run.
