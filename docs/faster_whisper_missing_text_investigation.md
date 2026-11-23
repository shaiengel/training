# faster-whisper missing text investigation

## Problem

Transcribing the lecture `33626.mp3` (Hebrew, about 18 minutes) with faster-whisper loses short stretches of text, from a few words to a few sentences. The audio is clear. Decoding a lost stretch on its own (for example `clip_timestamps="253,262"`) returns the missing words, so the model can transcribe it.

There are three separate causes, and they stack:

1. **Compression-ratio fallback.** The default threshold 2.4 is too low for Hebrew. The decoder retries at random temperatures and keeps a random sample.
2. **Token limit.** The decoder has 448 positions, and faster-whisper fills about half of them with previous text. Dense Hebrew windows run out of room and the output is cut off. Verified with logs (section "Cause 2").
3. **Early close.** Sometimes the model writes a closing timestamp and stops while speech remains in the window. faster-whisper trusts it and jumps a full 30s window. Verified with one example (window 612.58s); why it happens is unknown.

Causes 2 and 3 both end in the same way: faster-whisper starts the next window 30s later and the audio between the last decoded word and the window end is never decoded.

## How faster-whisper picks the next window

faster-whisper decodes audio in 30s windows. After each window it must decide where the next one starts. It decides by reading the timestamp tokens the model wrote.

### Timestamp tokens

The model writes a segment as `<|start|> text <|end|>`. Times are in seconds from the window start. When two segments follow each other, the end of one and the start of the next are written next to each other, so two timestamp tokens appear in a row:

```
<|0.00|> first sentence <|4.20|><|4.20|> second sentence <|9.00|><|9.00|> third sentence <|14.50|>
```

The code looks at how the token sequence ends (`_split_segments_by_timestamps`, `transcribe.py:1034-1101`). There are three ways. The names in the first column are what my `[window]` log prints. `single` is the case I sometimes call "closed".

| Log name | How the tokens end | What it means | Where the next window starts |
|---|---|---|---|
| `open` | `... <|9.00|><|9.00|> unfinished text` (ends in text after two timestamps in a row, or in two timestamps) | the last segment started but was cut or did not fit | at the last closed timestamp. The unfinished segment is dropped and decoded again. **Nothing is lost.** |
| `single` ("closed") | `... text <|14.50|>` (text, then one timestamp) | the model finished its last segment and says nothing more follows | **a full 30s later** |
| `collapsed` | no two timestamps in a row, and the tokens do not end text then timestamp. Usually one opening timestamp followed by all the text, with no closing timestamp | no usable end time. The window is one segment | **a full 30s later** |

Examples with a window from 0s to 30s and speech until 30s:

```
open:       <|0.00|> a <|10.0|><|10.0|> b <|20.0|><|20.0|> c (cut here)
            -> restarts at 20.0s, c is decoded again, nothing lost

single:     <|0.00|> a <|10.0|><|10.0|> b <|27.0|>
            -> model says it is done at 27.0s
            -> next window starts at 30s, speech from 27.0s to 30s is lost

collapsed:  <|0.00|> a b c d e f (cut here, no closing timestamp)
            -> no end time at all, next window starts at 30s
```

The jump for `single` and `collapsed` is `seek += segment_size` (`transcribe.py:1071` and `1101`). The code assumes that a closed last segment means silence follows. That is wrong when the model stopped early or was cut off right after a closing timestamp.

### What `word_timestamps=True` does and does not do

With `word_timestamps=True`, faster-whisper aligns every word to the audio and sets `seek` to the last word's end, but only when the window did not end as `single` (`transcribe.py:1288`, `if not single_timestamp_ending`). For `single` windows the full-window jump stays. So `word_timestamps=True` does not fix `single` windows. Earlier notes claimed it always moved `seek` to the last word. That was wrong.

## Cause 1: compression-ratio fallback

The default `compression_ratio_threshold` is 2.4. Hebrew text often exceeds it, so many windows retry at temperatures 0.2, 0.4, 0.6, 0.8 and 1.0. After the last retry it keeps the temperature-1.0 sample, which is random.

- Window ratios seen: 2.06 to 2.8. Normal Hebrew text, not repetition loops.
- Each fallback above 0.5 also resets the previous-text prompt (`Reset prompt` in the debug log).
- A window with fallbacks takes about 25s instead of about 4s.
- With `compression_ratio_threshold=2.6`, two windows of `33626.mp3` still fell back (2.73 at 81.18s and 2.65 at 787.88s). The highest ratio seen on a fine-tuned model was 2.74, so `2.8` or `None` avoids all fallbacks. `2.8` keeps a safety net against real repetition loops.
- Why Hebrew trips the threshold is an assumption, not tested: Hebrew letters take two bytes in UTF-8 and the first byte is always `0xD7`, so the text compresses well. The 2.4 default was tuned on English.
- A CUDA out-of-memory error occurred once on the first retry after a ratio failure. Retries above temperature 0 decode `best_of=5` sequences, which uses more VRAM. Not verified.

Status: `run_whisper.py` uses 2.6. `engines/faster_whisper_engine.py` (used by `run_bench.py`) and `C:\portal\transcription\gpu_instance\src\gpu_instance\services\transcriber.py` use the default, so they have the same exposure.

## Cause 2: token limit

### What the logs show

The decoder is limited to `max_length = 448` positions, and that count includes the prompt (`transcribe.py:722`). When `condition_on_previous_text` is on, faster-whisper puts up to 223 tokens of previous text in the prompt (`transcribe.py:1550`), plus `<|startofprev|>` and the 3 prefix tokens. Every window in the run had `prompt_tokens=227`, so the output budget is `448 - 227 = 221` tokens. Dense Hebrew takes about 2.8 tokens per word, so a 30s window with 77 to 80 words needs about 216 to 222 tokens, which is the whole budget.

Windows decoded with the full 223-token prompt (`prompt_tokens=227`, `budget=221`), fine-tuned model unless noted:

| Window | Ending | `output_tokens` | What happened |
|---|---|---|---|
| 198.18s | `collapsed` | 222 | text ends mid-sentence ("…זה לא"); next window 30s later |
| 226.88s | `single` | 220 | closed at 255.14s, 1.74s before the window end; "כלומר, כן, אפשר לקחת" lost |
| 256.88s | `open` | 222 | 12s undecoded, recovered by the next window |
| 220.36s (base) | `open` | 222 | recovered |

Every window that lost audio was at the budget (within one token). The base model also reaches the budget, but it writes sentence-level segments, so the cut lands in a short unfinished segment (`open`) and is decoded again. The fine-tuned model often writes one long segment, so a cut leaves no timestamp to restart from (`collapsed`), or lands right after a closing timestamp (`single`). The `single` explanation is an inference, not shown.

Token counts for the failing windows, measured with the fine-tuned model's tokenizer on the logged text (approximate, timestamps not counted): 221 tokens for the 447.08s window, 222 for 198.04s, 216 for 226.88s.

The clip run (`clip_timestamps="226.88,300"`) did not reproduce the loss because it starts with no previous text, so the full 448 positions were free for output.

### The `max_new_tokens` option

`max_new_tokens` does not help. It sets `max_length = len(prompt) + max_new_tokens` and raises an error if that exceeds 448 (`transcribe.py:193-206`). It can only lower the limit. Nothing in faster-whisper trades prompt length for output length.

### Fix: shorter previous-text prompt

`limit_previous_tokens(model, max_tokens=128)` in `run_whisper.py` caps the previous text at 128 tokens by wrapping `get_prompt`. The prompt becomes 132 tokens and the budget 316, above the 271 tokens at p99 of the training targets. This mirrors the preparator, which shortens the previous text to fit the target (`preperator.py:270-282`).

Results with `prompt_tokens=132`, `budget=316` (same file, `apply_fix=False`):

| | Base | Fine-tuned |
|---|---|---|
| windows | 41 | 40 |
| `single` | 4 | 4 |
| `open` | 37 | 32 |
| `collapsed` | 0 | 4 |
| windows with more than 1s undecoded | 1 | 5 |

For comparison, the runs with `prompt_tokens=227`:

| | Base | Fine-tuned |
|---|---|---|
| windows | 42 | 41 |
| `single` | 4 | 6 |
| `open` | 37 | 30 |
| `collapsed` | 1 | 5 |
| windows with more than 1s undecoded | 4 | 8 |

`output_tokens` in the shortened runs is 203 to 224, well under 316. The 255s junction is intact in both. The shorter prompt reduced the number of failing windows, especially for the base model, but did not remove them.

## Cause 3: early close

The window at 612.58s (`ending=single`, `last_ts=640.64s`, `gap_to_window_end=1.94s`, `output_tokens=224`, `budget=316`) is not a token-limit cut. The model finished "…נזכיר שלמדנו אתמול שיש שני סוגים של הגרמה." and closed the segment at 640.64s with 1.94s of window left. The next window started at 642.58s, and its first segment begins at 643.2s with "ויש הגרמה שזה להתחיל במקום הנכון". The words "יש הגרמה שזה פשוט לשחוט שלא במקום" were lost in the skipped stretch.

The model stopped early while speech remained and faster-whisper trusted it. This case is only handled by the seek fix below. Why the model does this is unknown.

## The seek fix in `run_whisper.py`

`fix_window_seek(model)` wraps `model._split_segments_by_timestamps` and returns `single_timestamp_ending=False`, so the word-based `seek` step at `transcribe.py:1288` runs for every window. The next window starts at the last aligned word's end. It requires `word_timestamps=True`.

- **Verified with the fix on** (before the prompt limit existed): the 255s junction and the 473.8s junction were recovered.
- **Side effect found:** on the last window (1053.10s to 1074.66s) the model closed at 1074.44s, `seek` moved back, 0.22s of silence was decoded again, and the model hallucinated "תודה רבה." four times. A guard keeps the original behavior on the last window (`segment_size < nb_max_frames`). The guard was added after that run and has not been re-run.
- **Cost:** a window that ends in real silence is decoded again from the last word. `vad_filter=True` removes most of that.
- Not applied to `engines/faster_whisper_engine.py` or the transcriber service.

Current state of `run_whisper.py`: `word_timestamps=True`, `compression_ratio_threshold=2.6`. `apply_faster_whisper_fixes` turns on the 128-token previous-text cap and the seek fix by default. Options on the command line:

- `--max-previous-tokens N` (default 128, `0` keeps faster-whisper's 223)
- `--no-seek-fix`
- `--log-windows` (off by default, see below)

VAD is commented out.

## Logging in `run_whisper.py`

Logging is off by default. Run with `--log-windows` to turn it on. It also prints every decoded segment and the `faster_whisper` debug logs. The `WindowLog` class records the ending the model produced before `fix_window_seek` changes it, so the counts do not depend on whether the fix is on. Each window prints one line, for example:

```
[window] start=612.58s end=642.58s ending=single last_ts=640.64s gap_to_window_end=1.94s output_tokens=224 prompt_tokens=132 budget=316
```

- `ending` is `open`, `single` or `collapsed`, as in the table above.
- `last_ts` is the last timestamp token the model wrote. For `collapsed` windows only the opening timestamp exists, so `last_ts` is the window start and the gap prints about 30s. That value is an artifact, not a measurement.
- `output_tokens` is the number of tokens the model generated for the window. `prompt_tokens` is the prompt length. `budget` is `448 - prompt_tokens`. A window is cut off when `output_tokens` reaches `budget`.
- The summary at the end prints the counts and the full line of every `single` and `collapsed` window. `full_jump_with_gap>1.0s` counts those windows with more than 1s undecoded. It includes every `collapsed` window because of the artifact above.
- With the fix on, these numbers still describe what the model emitted. They do not measure lost audio.

## Base model vs fine-tuned model

Counts are in the tables above. The fine-tuned model has more `collapsed` windows (4 to 5 against 0 to 1). It is one file and one run per model, and the windows of the two runs do not line up, so this is thin evidence. The base model used here is `ivrit-ai/whisper-large-v3-turbo-ct2`, the fine-tuned model is `portal-whisper-turbu-v3-5000-cosine-ct2`.

The fine-tuned model writes long segments, often one segment of 20 to 28s, while its training data has short segments (next section). The base model writes sentence-level segments. That difference is why the same token cut hurts the fine-tuned model more.

## Dataset check

Two datasets were checked.

**`C:\portal\dataset_builder\output\dataset_split\train`** (source transcripts, 11,362 samples from 11 of 106 shards): 78.3% end with two timestamps (cross-over), 21.7% end with one closing timestamp, none ends in text without a timestamp. `create_dataset.py:219-254` builds cross-over slices on purpose, and `preperator.py:219-226` keeps their timestamps.

**`C:\portal\dataset_builder\output\dataset_processed_large_turbo\train`** (processed labels, 4,575 samples from 14 of 335 shards):

| Measure | Value |
|---|---|
| samples in timestamp mode | 100% (none uses `<|notimestamps|>`) |
| samples with a previous-text prefix | 49% (2,253). All of them contain timestamps |
| previous-text prefix length | mean 180 tokens, p90 218, max 224 |
| target length | mean 185 tokens, p90 233, p99 271, max 323 |
| total label length | max 448 (previous text is shortened to fit) |
| timestamp tokens per sample | mean 13.9, about 7 segments |
| segment duration | mean 3.5s, median 2.8s, p90 6.7s, max 29.8s |
| ends with two timestamps (cross-over) | 79% |
| ends with one closing timestamp | 21% |

Token ids used for large-v3: `<|startofprev|>` is 50362, `<|startoftranscript|>` is 50258, `<|notimestamps|>` is 50364, the first timestamp is 50365, end-of-text is 50257.

Conclusions:
- The training data is sentence-level with a timestamp pair every few seconds, and it has the cross-over case in most samples. I found nothing in the format that explains the long segments and early closes. Adding more short sentences is not supported by this data, because it already has short segments.
- The model does not reproduce this style in some lecture windows. Why is unknown. Ideas, none verified: not enough training steps for timestamp placement (5,000), the lecture's speaking style differing from the training audio, or the previous-text style in the prompt.
- Nothing "adds" timestamps at inference. The model must write them itself. faster-whisper only reads them.
- Training prefixes are shortened to leave room for the target, but faster-whisper did not do that at inference. The shorter prompt fix (above) removes that difference.

## Other settings tested

- `without_timestamps=True`: cuts the audio into hard 30s blocks and loses words at each boundary. The sampled training data has no `<|notimestamps|>` examples. Keep it `False`.
- `condition_on_previous_text=False`: made the problem disappear on a 2-minute crop. The cause is now clear: with no previous text, the output gets the full 448 positions. A shorter prompt gives most of that benefit and keeps some context.
- VAD: did not fix the missing text, because the speech is continuous.
- A different tokenizer or `vocabulary.json`: would not help and would break the model.

## transformers pipeline comparison (earlier notes, not re-checked)

On the full file, transformers lost no text, but it has repeated clauses inside one segment, character stutters ("נשוווה", "עודוד"), and the same kind of timestamp collapse with wrong timestamps but no loss. It is not a drop-in fix.

## Other defects seen, not investigated

- Words glued together inside segments, for example "עושיםדווקא", "מביאהאת", "ישרכלפי", "הזהלחתוך". Each costs two word errors in WER. It appears in both models, so it is not caused by fine-tuning alone. Cause unknown.
- Recurring word errors around 429s ("כלפי מדה" and similar) on hard audio.
- Some stretches of the fine-tuned output have no punctuation and long segments, which makes segment timestamps unusable for subtitles.

## Current conclusion

1. Raise the compression threshold to 2.8 or disable it. Required in every faster-whisper path.
2. A 223-token previous-text prompt leaves about 221 output tokens. Dense Hebrew windows are cut off at that limit. Capping the previous text at 128 tokens reduced failing windows but did not remove them.
3. The model also closes some windows early (window 612.58s). This is only covered by the seek fix (`apply_fix=True`).
4. The seek fix recovered the junctions checked. The last-window guard is untested.
5. The training data is sentence-level and has the cross-over case. It does not explain the long segments or early closes.

## Next steps

- Rerun `33626.mp3` with `--log-windows` and the defaults (seek fix on, 128-token cap). Check that the 612.58s, 255s and 473.8s junctions are recovered and the "תודה רבה" loop at the end is gone.
- Test whether the fine-tuned model reproduces the training style on training audio: take about 50 training samples, decode them exactly as faster-whisper does, and compare the segment count and timestamps with the labels. This separates a training problem from a problem with the lecture audio.
- Try 100 and 160 tokens for `limit_previous_tokens` and compare the counts.
- Port the fix, the prompt limit and the threshold to `engines/faster_whisper_engine.py` and the transcriber service.
- Look into the glued words.
