# faster-whisper missing text investigation

## Problem

When transcribing a 2-minute crop (`output.mp3`, taken from `33626.mp3`) with the fine-tuned model `portal-whisper-large-v3-13000-cosine-ct2` through faster-whisper, some sentences were missing. In one run, roughly 56s to 86s was lost. The same happened on the full file, with `beam_size=5` and float16, and with `beam_size=1` and int8. The audio itself is clear.

## Root cause

faster-whisper's default `compression_ratio_threshold` is 2.4. Hebrew text often exceeds it, so almost every 30s window failed the check and the decoder retried at higher temperatures (0.0, 0.2, 0.4, 0.6, 0.8, 1.0). After the last retry it keeps whatever the temperature-1.0 sample produced.

- Window ratios seen: 2.06 to 2.64 for the fine-tuned model. Normal Hebrew text, not repetition loops.
- Temperature 1.0 is random sampling, so output varies between runs. This is why the problem appeared and disappeared and why prints or flag changes seemed to matter.
- Each fallback above 0.5 also resets the previous-text prompt (`Reset prompt` in the debug log), which is why `condition_on_previous_text` appeared to matter.
- Each window took about 25s because of the retries. With the fallback gone, about 4s.
- Why Hebrew trips the threshold (assumption, not tested): Hebrew letters take two bytes in UTF-8 and the first byte is always `0xD7`, so the text compresses well. The 2.4 default was tuned on English. Dense speech with repeated terms (עוף, דאורייתא, סימנים) raises the ratio further.
- The base model `ivrit-ai/whisper-large-v3-ct2` has the same problem. On window 1 its ratio was 2.403, it fell back to temperature 1.0, and the sampled output dropped the opening words "מחלוקת אחרת,". It only looked fine because debug logging was off.
- A CUDA out-of-memory error occurred on the first retry after a ratio failure. As far as I recall, retries above temperature 0 decode `best_of=5` sequences, which uses more VRAM. Not verified, but it fits the timing of the crash.

## Fix

Raise or disable the threshold:

```python
compression_ratio_threshold=2.8   # or None to disable
```

Results:
- `None`: all windows at temperature 0.0, deterministic, complete text, about 4s per window.
- `2.8`: same result, and it keeps a safety net against real repetition loops. The highest ratio seen was 2.64.
- Applied so far: `run_whisper.py` only.
- Not yet changed: `engines/faster_whisper_engine.py` (used by `run_bench.py`, so benchmark WER currently includes the random retries) and `C:\portal\transcription\gpu_instance\src\gpu_instance\services\transcriber.py` (only sets `language` and `beam_size`, so it has the same exposure).

## Other settings tested

- `without_timestamps=True`: cuts the audio into hard 30s blocks and loses words at each boundary (for example at 30s, 60s and 90s). The sampled training data had no `<|notimestamps|>` examples, so this mode is probably out of distribution for the model. Keep it `False`. The default is `False` for `WhisperModel.transcribe` and `True` for `BatchedInferencePipeline.transcribe`.
- `condition_on_previous_text=False`: made the problem disappear, but only because it avoids the prompt reset that follows a fallback. It was not the cause.
- `word_timestamps=True`: also hid the problem. It sets `seek` to the last aligned word's end (`transcribe.py:1291`), so skipped audio gets decoded again. It does not fix the model. It is still useful for better segment timing (for example VTT output).
- VAD: did not fix the missing text.
- Different tokenizer or `vocabulary.json`: would not help and would break the model. The ratio is computed on decoded UTF-8 text, and token ids are tied to the model embeddings.

## Dataset check

Two of the 332 shards in `C:\portal\dataset_builder\output\resplit_processed\train` were decoded (about 480 samples). Findings:
- About 45% of samples have a previous-text prefix: `<|startofprev|>` + previous text + `<|startoftranscript|>`.
- Every previous-text prefix contains timestamp tokens, each restarting from `<|0.00|>`. This matches what faster-whisper builds in default mode, so there is no format mismatch.
- All samples are in timestamp mode. None has `<|notimestamps|>`. So the effective timestamp ratio in this dataset is 1.0, not the 0.5 default.
- Caveat: only two shards were checked, and it was not confirmed that this dataset trained `checkpoint-13000`.

## Why transformers did not show the problem

The transformers pipeline decodes greedily with no compression-ratio or log-probability fallback unless you pass them in `generate_kwargs`. It never retried at temperature 1.0. (Assumption, not tested.)

## Unresolved

- **Timestamp quality in the fine-tuned model.** In some windows (for example 26.8s to 27.0s) the model emits one segment with a 0.2s span for about 30s of text. `seek` then advances a full window, and the cut can land mid-word ("באותה מח" followed by "בק בדף כז"). Other windows (84s onward) give proper sentence-level segments. The base model produces sentence-level segments in all windows. The cause is not known. It may be training or data related. Not caused by the compression issue.
- Whether the base model's sentence-level timing is the target to match, and whether the training data needs a check for window-start timestamp behavior.
- The apparent missing spaces in some pasted output ("שםמדגיש", "ברידרבא") look like terminal or paste artifacts. Not confirmed.

## Debugging additions in `run_whisper.py`

`transcribe_faster_whisper` now enables `faster_whisper` DEBUG logging and prints `seek`, `temp`, `no_speech`, `avg_logprob` and `compression` for each segment. The debug lines "Compression ratio threshold is not met", "Reset prompt" and "Processing segment at" are the useful ones.
