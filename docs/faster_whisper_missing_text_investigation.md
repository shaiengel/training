# faster-whisper missing text investigation

## Problem

Transcribing the lecture `33626.mp3` (Hebrew, about 18 minutes) with faster-whisper loses short stretches of text, from a few words to a few sentences. The audio is clear. Two crops and the full file were tested, with `beam_size=5` and float16, and with `beam_size=1` and int8. Both the fine-tuned model `portal-whisper-large-v3-13000-cosine-ct2` and the base model `ivrit-ai/whisper-large-v3-ct2` are affected.

There are two separate causes, and they stack:

1. **Compression-ratio fallback** (the default threshold 2.4 is too low for Hebrew). It makes the decoder sample at random temperatures and drop text.
2. **Timestamp collapse.** The model sometimes emits one segment with a tiny time span (0.1 to 0.7s) that holds a full 30s window of text. faster-whisper then advances `seek` by a full window, and any speech the model did not transcribe before the window end is skipped.

Fixing cause 1 is necessary but not sufficient. Cause 2 is still open.

## Cause 1: compression-ratio fallback

faster-whisper's default `compression_ratio_threshold` is 2.4. Hebrew text often exceeds it, so almost every 30s window failed the check and the decoder retried at higher temperatures (0.0, 0.2, 0.4, 0.6, 0.8, 1.0). After the last retry it keeps whatever the temperature-1.0 sample produced.

- Window ratios seen: 2.06 to 2.8 for both models. Normal Hebrew text, not repetition loops.
- Temperature 1.0 is random sampling, so output varies between runs. This is why the problem appeared and disappeared and why prints or flag changes seemed to matter.
- Each fallback above 0.5 also resets the previous-text prompt (`Reset prompt` in the debug log), which is why `condition_on_previous_text` appeared to matter.
- Each window took about 25s because of the retries. With the fallback gone, about 4s.
- Why Hebrew trips the threshold (assumption, not tested): Hebrew letters take two bytes in UTF-8 and the first byte is always `0xD7`, so the text compresses well. The 2.4 default was tuned on English. Dense speech with repeated terms (עוף, דאורייתא, סימנים) raises the ratio further.
- The base model has the same problem. On window 1 of the 2-minute crop its ratio was 2.403, it fell back to temperature 1.0, and the sampled output dropped the opening words "מחלוקת אחרת,". It only looked fine because debug logging was off.
- A CUDA out-of-memory error occurred on the first retry after a ratio failure. As far as I recall, retries above temperature 0 decode `best_of=5` sequences, which uses more VRAM. Not verified, but it fits the timing of the crash.

Fix: raise or disable the threshold.

```python
compression_ratio_threshold=2.8   # or None to disable
```

- `None`: all windows stay at temperature 0.0, deterministic, about 4s per window.
- `2.8`: same behavior and it keeps a safety net against real repetition loops. The highest ratio seen on the fine-tuned model was 2.74.
- On the 2-minute crop this alone gave complete text. On the full file it did not (see cause 2).
- Applied so far: `run_whisper.py` only. A test run with the base model still had the default 2.4 active (the log showed `> 2.400000` fallbacks), so check that the setting is passed through on every model path.
- Not yet changed: `engines/faster_whisper_engine.py` (used by `run_bench.py`, so benchmark WER currently includes the random retries) and `C:\portal\transcription\gpu_instance\src\gpu_instance\services\transcriber.py` (only sets `language` and `beam_size`, so it has the same exposure).

## Cause 2: timestamp collapse and full-window seek

In some windows the model emits a single segment such as `[319.9s - 320.0s]` (0.1s span) holding about 30s of text. faster-whisper sees no usable end timestamp and advances `seek` by the full window. If the model stopped emitting text before the audio reached the window end, the speech in between is never decoded. The cut can also land mid-word or mid-sentence ("בש" followed by "גגה").

Observed cases:

| Run | Where | What was lost |
|---|---|---|
| Fine-tuned, threshold 2.8, no word timestamps | `seek=31986` gave `[319.9-320.0]`, next window at 349.86s | words at the junction between "אחרי שהמשנה כבר" and "למדנו שהמיקום…" |
| Fine-tuned, threshold 2.8, `word_timestamps=True` | `seek=40248` ended "…כלפי מדא", next window at 432.46s | "זה מה שבני רבי חייא יאמרו שהמשנה" |
| Base model, default threshold 2.4 | `seek=45394` was a temp=1.0 fallback and gave `[453.9-454.6]`, next window at 483.94s | "אפשר ללחוץ כלפי מטה" |

Other examples: `seek=2784`, `44714`, `50714`, `56714` on the fine-tuned model each show a span of 0.1 to 0.5s for a full window of text. The base model shows smaller collapses (`[342.0-355.4]`, `[822.8-838.9]`).

Findings:
- Compression is ruled out for the first two rows: every window in those runs decoded at temperature 0.0 with ratios 2.09 to 2.74.
- The base model also collapses timestamps, including the extreme 0.7s case above. So this is not caused by fine-tuning alone. Earlier notes said the base model gave sentence-level segments in every window. That was wrong. It did so in the windows checked at the time.
- Whether fine-tuning makes it more frequent is unknown. Only one full run per model was compared, and the base run also had the fallback active.
- The text is usually continuous across a collapsed segment, so words are lost only at the window end, not inside the segment.
- The second row's loss at 432.46s is not explained. Either the decoder skipped the first words of the next window, or the previous window dropped them. Not verified against the audio. The text "כלפי מדא" is garbled in the faster-whisper and transformers runs, so the audio around 429s is hard for the model.

## How `word_timestamps` changes `seek`

Based on the reference to `transcribe.py:1291` and my memory of the faster-whisper source. Not re-read in full.

- **Without `word_timestamps`:** timestamps are the model's own tokens (for example `<|319.90|>`). The model predicts them like any other token and faster-whisper reads them. The decoder still computes cross-attention, because that is how it reads the audio, but nothing uses it for timing. `seek` is derived from these tokens. If the model collapses a window into one segment with no usable end timestamp, `seek` advances a full 30s window.
- **With `word_timestamps=True`:** faster-whisper runs an extra pass after decoding. It takes the cross-attention weights from the decoder's alignment heads (a specific subset of attention heads) and runs DTW on them to align each word to a point in the audio. Word and segment times come from this alignment, and `seek` is set to the last aligned word's end (`transcribe.py:1291`).
- Effect: the next window starts right after the last word that was actually transcribed, so audio the model skipped at the window end is decoded again. Segments are also rebuilt from word timings, so a 30s blob is split into sentence-sized segments.
- Limits:
  - The model still decides what to transcribe. A window can still drop words.
  - The alignment is a post-hoc estimate from attention patterns, not something the model was trained to output. If it puts the last word too late (for example at the window end), `seek` skips audio again. This is a possible explanation for the loss at 432.46s on the fine-tuned model. Not verified.
  - The cut can land mid-word, which leaves a truncated fragment at the end of one window and the full word at the start of the next ("ללחו" then "כלפי", "אותו ד" then "דבר").
  - It is slower because of the extra alignment pass.

## Other settings tested

- `without_timestamps=True`: cuts the audio into hard 30s blocks and loses words at each boundary (for example at 30s, 60s and 90s). The sampled training data had no `<|notimestamps|>` examples, so this mode is probably out of distribution for the model. Keep it `False`. The default is `False` for `WhisperModel.transcribe` and `True` for `BatchedInferencePipeline.transcribe`.
- `condition_on_previous_text=False`: made the problem disappear on the 2-minute crop, but only because it avoids the prompt reset that follows a fallback. It was not the cause.
- `word_timestamps=True`: sets `seek` to the last aligned word's end (`transcribe.py:1291`), so skipped audio can be decoded again. It removed the loss at 349.86s on the fine-tuned model, but not the loss at 432.46s. It is a partial mitigation, not a fix. It is also useful for better segment timing (for example VTT output).
- VAD: did not fix the missing text.
- Different tokenizer or `vocabulary.json`: would not help and would break the model. The ratio is computed on decoded UTF-8 text, and token ids are tied to the model embeddings.

## Verification run: base model, threshold 2.8, `word_timestamps=True`

Full file, every window at temperature 0.0, ratios 2.15 to 2.69.

- No missing sentences. All window junctions are continuous.
- The spots that failed in earlier runs are present: "אפשר ללחוץ כלפי מטה" (lost at 453.9s in the base run with the default threshold) and "זה מה שבני רבי חייא יאמרו שהמשנה" (lost at 432.46s in the fine-tuned run).
- Remaining defects:
  - Words cut at window junctions leave a truncated fragment, then the full word in the next window ("ללחו", "אותו ד", "אפ"). This costs nothing for WER.
  - One substitution at 371.2s: "כיוון שלגבי שחיטה" became "כיוון … אם יש לגבי שחיטה".
  - Recurring word errors: "תפסול" for "הפסול" (which also drops "את"), "רוואחא", "רוואשי", and "כלפי מדע". The last one is the same hard audio around 429s that every run garbles.
  - From 232s on there is no punctuation and each window is one segment (for example `[232.1s - 260.5s]`). Before 232s the segments are sentence-level with punctuation. The fine-tuned run drifts the same way after about 402s. A likely cause is previous-text conditioning feeding punctuation-free text back in (a guess). This does not affect WER because evaluation strips punctuation, but the segment timestamps are not usable for subtitles or VTT output.
- The fine-tuned model with the same settings lost words at 432.46s. That hints the fine-tuned model is more prone to it, but it does not prove it. The window boundaries differ between the runs (428.9s in the base run, 432.46s in the fine-tuned run), so the base run may simply not have hit a bad boundary.

## transformers pipeline comparison

On the full file, transformers lost no text. In particular it kept "זה מה שבני רבי חייא יאמרו שהמשנה" and "אפשר ללחוץ כלפי מטה". But it has its own problems:

- **Repeats inside one segment**: the same clause twice in `[185.2s - 227.6s]`, and "או שחייבים קודם כל לחתוך את המפרקת" twice in `[228.0s - 259.2s]`. The faster-whisper run at threshold 2.8 has neither. Likely cause is greedy decoding with no fallback, and a single repeated clause barely raises the compression ratio. This is inferred, not tested.
- **Character stutters** such as "נשוווה", "עודוד", "כמומו", "מלליקה".
- **The same timestamp collapse**: `[497.8-498.1]`, `[514.1-514.4]`, `[612.6-613.0]`, `[918.3-918.6]` and a 125s span `[11.9s - 136.9s]`. Here it causes no loss, only wrong timestamps.
- The same garbled word "כלפי מדא" appears, which fits a hard stretch of audio.

Earlier notes said transformers decodes greedily with no fallback unless you pass it in `generate_kwargs`. That is still the assumption. Untested options for it: `num_beams=5`, a temperature tuple, `compression_ratio_threshold=1.35` (computed on token ids, not on text, so the faster-whisper values do not apply), `logprob_threshold=-1.0`, `condition_on_prev_tokens=False`.

## Dataset check

Two of the 332 shards in `C:\portal\dataset_builder\output\resplit_processed\train` were decoded (about 480 samples). Findings:
- About 45% of samples have a previous-text prefix: `<|startofprev|>` + previous text + `<|startoftranscript|>`.
- Every previous-text prefix contains timestamp tokens, each restarting from `<|0.00|>`. This matches what faster-whisper builds in default mode, so there is no format mismatch.
- All samples are in timestamp mode. None has `<|notimestamps|>`. So the effective timestamp ratio in this dataset is 1.0, not the 0.5 default.
- Caveat: only two shards were checked, and it was not confirmed that this dataset trained `checkpoint-13000`.

## Current conclusion

1. The compression threshold must be raised (2.8) or disabled. This removes the random temperature-1.0 output and is required in every faster-whisper path (`run_whisper.py` done; `engines/faster_whisper_engine.py` and the transcriber service not yet).
2. After that, short losses remain at window ends when the model emits a collapsed timestamp segment. This happens in the base model too, so it is Whisper long-form behavior that fine-tuning may or may not have worsened.
3. `word_timestamps=True` reduces these losses. With the base model it gave a complete transcript; with the fine-tuned model it still lost words once (see "Verification run"). It works because `seek` follows the DTW-aligned end of the last word instead of the model's timestamp tokens.
4. transformers does not lose text on this file but has repeats and stutters, so it is not a drop-in fix either.
5. Not yet known: how often each model collapses timestamps, and whether fine-tuning changes that.

## Next steps

- Done: base model with `compression_ratio_threshold=2.8` and `word_timestamps=True` was complete (see "Verification run"). This is only one run, so it does not show whether fine-tuning makes the problem worse.
- Shift the window boundaries by prepending 10s of silence and rerun both models with 2.8 and `word_timestamps=True`. If the losses move or vanish, they depend on where the boundary falls. If only the fine-tuned model loses words, fine-tuning is the cause.
- Check the training data for window-start timestamp behavior.
- Try transformers with beams, fallback and `condition_on_prev_tokens=False` to see if the repeats go away.
- The apparent missing spaces in some pasted output ("שםמדגיש", "ברידרבא") look like terminal or paste artifacts. Not confirmed.

## Debugging additions in `run_whisper.py`

`transcribe_faster_whisper` enables `faster_whisper` DEBUG logging and prints `seek`, `temp`, `no_speech`, `avg_logprob` and `compression` for each segment. The debug lines "Compression ratio threshold is not met", "Reset prompt" and "Processing segment at" are the useful ones.
