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
python train-whisper.py --output_model_name lr-prep --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --train_datasets <user>/daf-yomi-lr-subset:train --eval_datasets <user>/daf-yomi-lr-subset:test --save_processed /workspace/training/output/lr_prep --ds_processor_proc_num 4

uv run python train-whisper.py --output_model_name lr-1e-4 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --use_preprocessed /workspace/training/output/lr_prep --learning_rate 1e-4 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-1e-4 --mixed_precision bf16

uv run python train-whisper.py --output_model_name lr-5e-5 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --use_preprocessed /workspace/training/output/lr_prep --learning_rate 5e-5 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-5e-5 --mixed_precision bf16

uv run python train-whisper.py --output_model_name lr-3e-5 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --use_preprocessed /workspace/training/output/lr_prep --learning_rate 3e-5 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-3e-5 --mixed_precision bf16

uv run python train-whisper.py --output_model_name lr-1e-5 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --use_preprocessed /workspace/training/output/lr_prep --learning_rate 1e-5 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-1e-5 --mixed_precision bf16

uv run python train-whisper.py --output_model_name lr-5e-6 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --use_preprocessed /workspace/training/output/lr_prep --learning_rate 5e-6 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-5e-6 --mixed_precision bf16
```

### Eval loss by epoch

| Epoch | 1e-4  | 5e-5  | 3e-5  | 1e-5  | 5e-6  |
|-------|-------|-------|-------|-------|-------|
| 0.2   | 0.574 | 0.488 | 0.458 | 0.446 | 0.459 |
| 0.4   | 0.586 | 0.483 | 0.440 | 0.419 | 0.423 |
| 0.6   | 0.567 | 0.469 | 0.434 | 0.407 | 0.411 |
| 0.8   | 0.553 | 0.474 | 0.427 | 0.400 | 0.405 |
| 1.0   | 0.548 | 0.449 | 0.420 | 0.395 | 0.397 |

### Final train loss

| LR   | 1e-4  | 5e-5  | 3e-5  | 1e-5  | 5e-6  |
|------|-------|-------|-------|-------|-------|
| Loss | 0.569 | 0.480 | 0.452 | 0.438 | 0.451 |

### Analysis

1e-5 wins: lowest eval loss (0.395), still dropping at epoch 1.0, no signs of plateau. Higher LRs overshoot — the pretrained model benefits from gentle fine-tuning. 5e-6 is close but converges slightly slower; 1e-4 and 5e-5 flatten early.

Note: batch size here is 4 × 4 = 16 (same effective batch as section 4). If you change the batch size for the real run, the LR may need re-tuning.

## 6. The real run

Use the full dataset with the chosen LR. Keep the eval set small:

- Pass `--eval_datasets portal-daf-yomi/portal-daf-yomi-whisper-training:test[:500]` and `--max_eval_set_size 500`.
- `--max_eval_set_size` is applied only after the whole eval set has been preprocessed. Slicing the spec as above avoids preprocessing all 8.88K test rows.
- The dataset has no validation split. Use rows from `test` for the small eval set, and keep a separate set of `test` rows that never takes part in training or checkpoint selection for the final score.
- Reset `--warmup_steps`, `--gradient_accumulation_steps` and `--logging_steps` for the longer run.
