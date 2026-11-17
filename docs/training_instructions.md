# Training Instructions

## Step 1: Preprocess Dataset

```bash
uv run python train-whisper.py --output_model_name prep --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3-turbo --train_datasets C:\portal\dataset_builder\output\dataset_split:train --eval_datasets C:\portal\dataset_builder\output\dataset_split:test --save_processed C:\portal\dataset_builder\output\dataset_processed_large_turbo --ds_processor_proc_num 1 --include_timestamps_prob 1.0
```

### Why preprocess?

The raw dataset contains audio files and text transcripts. Whisper training requires a different format: Mel spectrograms (not raw audio), tokenized label sequences (not plain text), and training-specific augmentations like timestamp token injection and previous-text conditioning. This preprocessing step converts the dataset into that format.

This is different from the Hugging Face dataset format, which stores raw audio arrays and plain text. The HF format is model-agnostic — it doesn't know which tokenizer, feature extractor, or training strategy you'll use. Preprocessing bakes in the specific model's vocabulary, Mel parameters, and training decisions (e.g., always include timestamps), producing tensors ready for the training loop.

Saving the preprocessed result with `--save_processed` avoids repeating this work on future training runs.

### Parameters

- **`--output_model_name prep`** — Placeholder name. This run is for preprocessing only, not actual training.
- **`--skip_push_to_hub`** — Don't upload anything to Hugging Face Hub.
- **`--model_name ivrit-ai/whisper-large-v3-turbo`** — The base model. Its tokenizer and feature extractor are needed to preprocess the dataset (tokenize transcripts, extract Mel spectrograms).
- **`--train_datasets ... :train`** — Path to the local dataset, using the `train` split.
- **`--eval_datasets ... :test`** — Path to the local dataset, using the `test` split.
- **`--save_processed`** — Save the preprocessed dataset to disk so future training runs can skip preprocessing via `--use_preprocessed`.
- **`--ds_processor_proc_num 1`** — Single-process preprocessing. Increase for faster processing on machines with more CPU cores.
- **`--include_timestamps_prob 1.0`** — Train with timestamps on 100% of samples. Set to 1.0 because inference will always run with `without_timestamps=False`. Default is 0.5, which strips timestamps from some samples to support both modes — unnecessary here.

## Step 2: Learning Rate Search

Run short training runs (125 steps) with different learning rates to find the best one. Compare loss curves to pick the rate that decreases loss fastest without instability.

```bash
uv run python train-whisper.py --output_model_name lr-1e-4 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --use_preprocessed /workspace/training/output/lr_prep --learning_rate 1e-4 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-1e-4 --mixed_precision bf16

uv run python train-whisper.py --output_model_name lr-5e-5 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --use_preprocessed /workspace/training/output/lr_prep --learning_rate 5e-5 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-5e-5 --mixed_precision bf16

uv run python train-whisper.py --output_model_name lr-3e-5 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --use_preprocessed /workspace/training/output/lr_prep --learning_rate 3e-5 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-3e-5 --mixed_precision bf16

uv run python train-whisper.py --output_model_name lr-1e-5 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --use_preprocessed /workspace/training/output/lr_prep --learning_rate 1e-5 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-1e-5 --mixed_precision bf16

uv run python train-whisper.py --output_model_name lr-5e-6 --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 --use_preprocessed /workspace/training/output/lr_prep --learning_rate 5e-6 --max_steps 125 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --warmup_steps 10 --logging_steps 5 --eval_steps 25 --save_steps 100000 --run_name lr-5e-6 --mixed_precision bf16
```

### Why search for learning rate?

Too high and training diverges (loss spikes or oscillates). Too low and training is slow or gets stuck. A short sweep over a range (5e-6 to 1e-4) with identical settings reveals which rate gives the steepest stable loss decrease.

### Parameters

- **`--use_preprocessed`** — Load the preprocessed dataset from Step 1 instead of re-processing from raw data.
- **`--learning_rate`** — The rate being tested. Five values are swept: 1e-4, 5e-5, 3e-5, 1e-5, 5e-6.
- **`--max_steps 125`** — Short run, enough to see the loss trend without wasting compute.
- **`--per_device_train_batch_size 4`** — Samples per GPU per step.
- **`--gradient_accumulation_steps 4`** — Accumulate 4 steps before updating weights. Effective batch size = 4 × 4 = 16.
- **`--warmup_steps 10`** — Brief warmup to avoid early instability.
- **`--logging_steps 5`** — Log loss every 5 steps for a detailed curve.
- **`--eval_steps 25`** — Run evaluation every 25 steps to track validation loss.
- **`--save_steps 100000`** — Set higher than max_steps to skip saving checkpoints during the sweep.
- **`--run_name`** — Labels each run in the tracker (e.g., Weights & Biases) for easy comparison.
- **`--mixed_precision bf16`** — Use bfloat16 for faster training and lower memory usage.
