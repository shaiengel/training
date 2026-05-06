# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

When asked about installation, setup, or deployment (including vast.ai), refer to the Installation section in README.md as the source of truth.

## Commands

```bash
# Install dependencies
uv sync

# Run all tests
pytest tests/

# Run a single test
pytest tests/create_dataset/test_generate_slices.py::test_generate_slices

# Format code
black . --line-length 120
isort . --profile black

# Fix CUDNN path for faster-whisper (if libcudnn_ops error appears)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```

## Architecture

This is a Hebrew ASR (Automatic Speech Recognition) training and evaluation framework for Whisper models, built for ivrit.ai.

**Main scripts** (all at the root level):
- `train-whisper.py` — Fine-tune Whisper with optional QLoRA; handles dataset loading, preprocessing, training, and HF Hub upload
- `create_dataset.py` — Convert raw audio + JSON transcripts into HF datasets using stable-whisper for segmentation
- `run_bench.py` — Run the full evaluation suite across multiple datasets and engines
- `evaluate_model.py` — Evaluate a single model; computes WER, WIL, and character-level metrics with Hebrew text normalization
- `run_whisper.py` — Single-file inference runner
- `generate_model_formats.py` — Convert HF models to CT2/ONNX/GGML formats
- `merge-lora-whisper.py` — Merge LoRA adapters into base model weights

**Inference engine plugin system** (`engines/`):
Each engine module exposes `create_app(**kwargs) -> Callable`. The callable takes an entry dict `{"audio": {"array": np.ndarray, "sampling_rate": int}}` and returns `(transcription_text, elapsed_seconds)`. Engines are loaded dynamically by path, so adding a new backend requires no changes outside the new file.

**Data preprocessing** (`preprocess/`):
`DatasetPreparator` in `preperator.py` handles resampling to 16kHz, Mel-spectrogram extraction, timestamp token injection, conditioning on previous text, audio shift augmentation, and delta-encoded padding storage for disk efficiency. Key probabilities: `timestamp_sample_prob=0.5`, `condition_on_prev_sample_prob=0.5`.

**Training details**:
- QLoRA config: rank=64, alpha=1, RSLoRA, targets `q_proj, k_proj, v_proj, fc1, fc2, out_proj`, dropout=0.05
- Custom `compute_loss_func()` works around a gradient accumulation bug in Whisper's Transformers loss
- Dataset slicing uses HF ReadInstruction syntax (e.g., `dataset[:24634]`)

**Hebrew text normalization** (used in evaluation):
Strips RTL control characters, niqqud (diacritical marks), quotes, and punctuation before WER computation; applied on top of Whisper's `BasicTextNormalizer`.

**Evaluation datasets**: ivrit-ai/eval-d1, upai-inc/saspeech, google/fleurs (he_il), mozilla-foundation/common_voice_17_0 (gated), imvladikon/hebrew_speech_kan — all Hebrew, all using the `test` or `validation` split.
