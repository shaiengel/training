# faster-whisper missing text investigation

## Problem

Transcribing the lecture `33626.mp3` (Hebrew, about 18 minutes) with faster-whisper loses short stretches of text, from a few words to a few sentences. The audio is clear. Decoding a lost stretch on its own (for example `clip_timestamps="253,262"`) returns the missing words, so the model can transcribe it.

There are three separate causes, and they stack:

1. **Compression-ratio fallback.** The default threshold 2.4 is too low for Hebrew. The decoder retries at random temperatures and keeps a random sample.
2. **Token limit.** CTranslate2 caps the output of each window at `min(224, 448 - prompt_length)` tokens. Dense Hebrew windows reach the cap and the output is cut off. See "Cause 2" and the correction in "Root cause".
3. **Early close.** Sometimes the model writes a closing timestamp and stops while speech remains in the window. faster-whisper trusts it and jumps a full 30s window. Two examples were recorded (windows 226.88s and 612.58s), but the 612.58s window has `output_tokens=224`, so it was probably a cut at the cap, not an early close (see "Root cause").

Causes 2 and 3 can both end the same way: the window ends as `single` (below), faster-whisper starts the next window 30s later, and the audio between the last decoded word and the window end is never decoded. With `word_timestamps=True`, only `single` windows lose audio. `collapsed` and `open` windows are recovered. With `word_timestamps=False`, `collapsed` windows also jump a full window and lose audio. The problem reproduces better that way.

## Root cause (2026-11-24)

These findings come from later runs of `33626.mp3` with the fine-tuned model `whisper-ivrit-talmud-large-v3-turbo-ct2` (fine-tuned from `ivrit-ai/whisper-large-v3-turbo`, trained on `dataset_processed_large_turbo`), beam size 5, `word_timestamps=False`, 128-token previous-text cap. They correct parts of the sections below.

### The output cap is 224 tokens, not `448 - prompt`

CTranslate2 sets the decoding limit to `min(max_length / 2, max_length - prompt_length)` (`src/models/whisper.cc`, v4.5.0). faster-whisper passes `max_length=448`, so a window can never produce more than 224 tokens, whatever the prompt length. Tested on the 27.84s window: prompts of 3, 54, 132 and 204 tokens all stopped at exactly 224 tokens. OpenAI Whisper uses the same limit (`n_text_ctx // 2`).

So the 128-token previous-text cap raises the budget from 221 to 224, not to 316. That is why it reduced the failing windows only a little. Every "budget 316" below should read 224.

### How the audio is lost

In the beam-5 run (42 windows: 35 `open`, 6 `collapsed`, 1 `single`), every window that lost audio was `collapsed` and had exactly `output_tokens=224`: 198.20s, 370.04s, 612.62s, 914.06s, 1001.60s and 1044.52s. In each one:

1. The model wrote one opening timestamp and then only text, 30s of speech with no segment boundary.
2. The window reached the 224-token cap in the middle of a word (for example "זה לא חי|" at 198.20s).
3. With no timestamp to restart from, faster-whisper jumped a full 30s, and the rest of the window was never decoded.

The one `single` window (168.20s) closed 0.12s before the window end and lost nothing. No window ended early by the model's own choice in this run.

When a window is cut off but has written two timestamps in a row, it ends `open` and the next window decodes the cut text again. 9 of the 35 `open` windows hit 224 tokens and lost nothing.

### Why the model writes no boundary

The model writes a segment boundary only where the speaker pauses for about 0.3s or more:

- Every boundary it writes on `33626.mp3` has a gap of 0.28s or more between segment end and next start (30 boundaries, median 0.36s).
- The lecture has almost no such pauses: no 30s window has a pause over 0.45s.
- So it writes mostly 3 timestamps per window (one boundary), and sometimes only the opening one.

It learned this from the training labels. The dataset builder merges neighbouring segments when the gap between them is under 0.3s (`_merge_slice_segments` in `C:\portal\dataset_builder\src\dataset_builder\infrastructure\whisper_dataset_generator.py:286`, same code in `create_dataset.py:301`). Of 36,557 boundaries in `dataset_processed_large_turbo`, none has a gap of 0.2s or less. In seven stable-ts source transcripts, the merge removes 21% to 59% of the real sentence ends. The change is described in `create_dataset_segment_merge_change.md`.

### What was ruled out

- **Timestamp prediction in general:** on 80 eval samples of the training dataset, the fine-tuned model writes 5.8 boundaries per window, the labels 6.4. The base model writes 6.5.
- **The previous-text prompt:** with `condition_on_previous_text=False` the lecture still gets mostly 3 timestamps per window and 4 `collapsed` windows.
- **Decoding overriding the model:** CTranslate2 applies OpenAI's rule that forces a timestamp when all timestamps together beat the best text token (`ApplyTimestampRules`). The model itself preferred text.
- **The audio:** windows that lost audio and normal windows have the same pause statistics.
- **Long segments in the training targets:** the labels have about 7 segments per 30s. The problem is where those boundaries are, not how many.

### Base model on the same lecture

`ivrit-ai/whisper-large-v3-turbo-ct2` writes mostly 5 to 9 timestamps per window (up to 15) and has 2 `collapsed` windows. The fine-tuned model writes mostly 3 and has 6. Both hit the 224-token cap 15 times. The base model splits at sentence ends without pauses, so more of its cut windows have a point to restart from.

### Note on `--show-logits`

`--show-logits` forces greedy decoding (CTranslate2 returns logits only for `beam_size=1`). That can change the result: in one run greedy decoding lost a sentence at 52.44s that beam size 5 decodes correctly. Check that a lossy window is also lossy without `--show-logits` before reading its logits.

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
| `collapsed` | no two timestamps in a row, and the tokens do not end text then timestamp. Usually one opening timestamp followed by all the text, with no closing timestamp | no usable end time. The window is one segment | **without `word_timestamps`: a full 30s later.** With `word_timestamps=True`: at the last aligned word's end. Nothing is lost |

Examples with a window from 0s to 30s and speech until 30s:

```
open:       <|0.00|> a <|10.0|><|10.0|> b <|20.0|><|20.0|> c (cut here)
            -> restarts at 20.0s, c is decoded again, nothing lost

single:     <|0.00|> a <|10.0|><|10.0|> b <|27.0|>
            -> model says it is done at 27.0s
            -> next window starts at 30s, speech from 27.0s to 30s is lost

collapsed:  <|0.00|> a b c d e f (cut here, no closing timestamp)
            -> no end time at all
            -> without word_timestamps the next window starts at 30s
            -> with word_timestamps=True it starts at the end of word f
```

Both `single` and `collapsed` set `seek += segment_size` first (`transcribe.py:1071` and `1101`). The code assumes that a closed last segment means silence follows. That is wrong when the model stopped early or was cut off right after a closing timestamp.

### What `word_timestamps=True` does and does not do

With `word_timestamps=True`, faster-whisper aligns every word to the audio and sets `seek` to the last word's end, but only when the window did not end as `single` (`transcribe.py:1288`, `if not single_timestamp_ending`).

- `collapsed` and `open` windows: the step runs, so the next window starts at the last decoded word and nothing is lost. Checked in the log: an unpatched run had window 198.18s end `collapsed` and the next window start at 226.88s, the end of the last word, not at 228.18s.
- `single` windows: the step is skipped, so the full-window jump stays. This is the only ending that loses audio, and the only one the seek fix changes.
- Without `word_timestamps`, `collapsed` windows also jump a full window. That is what happened in the first runs, for example at `seek=44714`.

Earlier notes claimed that `word_timestamps=True` always moves `seek` to the last word, and later notes claimed `collapsed` windows jump a full window even with it. Both were wrong.

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
| 198.18s | `collapsed` | 222 | text ends mid-sentence ("…זה לא"); next window starts at 226.88s (last word end). Nothing lost |
| 226.88s | `single` | 220 | closed at 255.14s, 1.74s before the window end; "כלומר, כן, אפשר לקחת" lost |
| 256.88s | `open` | 222 | 12s undecoded, recovered by the next window |
| 220.36s (base) | `open` | 222 | recovered |

All four windows are at the budget (within one token), so they were cut off. What the cut costs depends on how the cut output ends:

- Cut inside an unfinished segment (`open`): the segment is decoded again. Nothing lost.
- Cut inside one long segment with no closing timestamp (`collapsed`): with `word_timestamps=True`, the next window starts at the last word. Nothing lost.
- Cut right after a closing timestamp (`single`): the next window starts 30s later. Audio is lost. This is the only window of the four that lost audio (226.88s).

That a cut can land right after a closing timestamp is an inference, not shown. The `single` windows with a gap could also be early closes by the model, as in the 612.58s example.

Token counts for the failing windows, measured with the fine-tuned model's tokenizer on the logged text (approximate, timestamps not counted): 221 tokens for the 447.08s window, 222 for 198.04s, 216 for 226.88s.

The clip run (`clip_timestamps="226.88,300"`) did not reproduce the loss. The earlier explanation, that with no previous text the full 448 positions were free for output, is wrong: the output cap is 224 tokens with any prompt (see "Root cause"). Without the prompt, the budget rises only from 221 to 224, and the clip also starts its windows at different times.

### The `max_new_tokens` option

`max_new_tokens` does not help. It sets `max_length = len(prompt) + max_new_tokens` and raises an error if that exceeds 448 (`transcribe.py:193-206`). It can only lower the limit. Nothing in faster-whisper trades prompt length for output length.

### Fix: shorter previous-text prompt

`limit_previous_tokens(model, max_tokens=128)` in `run_whisper.py` caps the previous text at 128 tokens by wrapping `get_prompt`. The prompt becomes 132 tokens. This was expected to raise the budget to 316, above the 271 tokens at p99 of the training targets, but CTranslate2 caps the output at 224 tokens (see "Root cause"), so the budget rises only from 221 to 224. This mirrors the preparator, which shortens the previous text to fit the target (`preperator.py:270-282`).

Results with `prompt_tokens=132` (real budget 224; these runs printed 316) (same file, `apply_fix=False`):

| | Base | Fine-tuned |
|---|---|---|
| windows | 41 | 40 |
| `single` | 4 | 4 |
| `open` | 37 | 32 |
| `collapsed` | 0 | 4 |
| `single` or `collapsed` with more than 1s gap (see note below) | 1 | 5 |

For comparison, the runs with `prompt_tokens=227`:

| | Base | Fine-tuned |
|---|---|---|
| windows | 42 | 41 |
| `single` | 4 | 6 |
| `open` | 37 | 30 |
| `collapsed` | 1 | 5 |
| `single` or `collapsed` with more than 1s gap (see note below) | 4 | 8 |

Note on the last row: with `word_timestamps=True`, only `single` windows lose audio, but these runs counted `collapsed` windows in that row too. The `single` counts are the relevant ones: base 4 and fine-tuned 6 with the full prompt, base 4 and fine-tuned 4 with the capped prompt. A `single` window loses audio only when its gap is large, so not every `single` window loses something.

`output_tokens` in the shortened runs is 203 to 224. This was read as well under the budget, but 224 is the real cap, so the windows at 224 were cut off. The 255s junction is intact in both. The shorter prompt changed the `single` and `collapsed` counts (fine-tuned 6 and 5 to 4 and 4, base 4 and 1 to 4 and 0), but did not remove them.

## Cause 3: early close

**Correction:** this window has `output_tokens=224`, which is the real output cap (see "Root cause"). It was probably cut off right after a closing timestamp, which looks exactly like an early close. The original analysis follows.

The window at 612.58s (`ending=single`, `last_ts=640.64s`, `gap_to_window_end=1.94s`, `output_tokens=224`, `budget=316`) is not a token-limit cut. The model finished "…נזכיר שלמדנו אתמול שיש שני סוגים של הגרמה." and closed the segment at 640.64s with 1.94s of window left. The next window started at 642.58s, and its first segment begins at 643.2s with "ויש הגרמה שזה להתחיל במקום הנכון". The words "יש הגרמה שזה פשוט לשחוט שלא במקום" were lost in the skipped stretch.

The model stopped early while speech remained and faster-whisper trusted it. This case is only handled by the seek fix below. Why the model does this is unknown.

## The seek fix in `run_whisper.py`

`fix_window_seek(model)` wraps `model._split_segments_by_timestamps` and returns `single_timestamp_ending=False`, so the word-based `seek` step at `transcribe.py:1288` also runs for `single` windows. The next window starts at the last aligned word's end. It requires `word_timestamps=True`. It changes only `single` windows, since `collapsed` and `open` windows already take that path upstream.

- **Verified with the fix on** (before the prompt limit existed): the 255s junction was recovered (next window at 255.12s). The 473.8s junction was also recovered, but that window ended `collapsed`, so `word_timestamps=True` alone recovers it and it is not evidence for the fix.
- **Side effect found:** on the last window (1053.10s to 1074.66s) the model closed at 1074.44s, `seek` moved back, 0.22s of silence was decoded again, and the model hallucinated "תודה רבה." four times. A guard keeps the original behavior on the last window (`segment_size < nb_max_frames`). The guard was added after that run and has not been re-run.
- **Cost:** a window that ends in real silence is decoded again from the last word. `vad_filter=True` removes most of that.
- Not applied to `engines/faster_whisper_engine.py` or the transcriber service.

Current state of `run_whisper.py`: `compression_ratio_threshold=2.6`. `word_timestamps` is on only with `--show-words`, so by default the seek fix has no effect and `collapsed` windows jump a full window, as upstream. `apply_faster_whisper_fixes` turns on the 128-token previous-text cap and the seek fix by default. Options on the command line:

- `--max-previous-tokens N` (default 128, `0` keeps faster-whisper's 223)
- `--no-seek-fix`
- `--log-windows` (off by default, see below)
- `--show-words`: print every word with its start, end and probability. Turns on `word_timestamps`, which hides the loss of `collapsed` windows.
- `--show-logits K`: print the top-K candidates and log-probabilities at every decoding step. Forces greedy decoding (see "Root cause").
- `--output FILE`: write the output to a UTF-8 file. Use it instead of `>` in Windows PowerShell, which corrupts the Hebrew.

Segment timestamps are always printed.

VAD is commented out.

## Logging in `run_whisper.py`

Logging is off by default. Run with `--log-windows` to turn it on. It also prints every decoded segment and the `faster_whisper` debug logs. The `WindowLog` class records the ending the model produced before `fix_window_seek` changes it, so the counts do not depend on whether the fix is on. Each window prints one line, for example:

```
[window] start=612.62s end=642.62s ending=collapsed last_ts=613.00s gap_to_window_end=29.62s output_tokens=224 prompt_tokens=132 budget=224
```

- `ending` is `open`, `single` or `collapsed`, as in the table above.
- `last_ts` is the last timestamp token the model wrote. For `collapsed` windows only the opening timestamp exists, so `last_ts` is the window start and the gap prints about 30s. That value is an artifact, not a measurement.
- `output_tokens` is the number of tokens the model generated for the window. `prompt_tokens` is the prompt length. `budget` is `min(224, 448 - prompt_tokens)`, the CTranslate2 cap. Runs before 2026-11-24 printed `448 - prompt_tokens`, which was wrong. A window is cut off when `output_tokens` reaches `budget`.
- The summary at the end prints the counts and the full line of every window that can lose audio: `single` windows, and also `collapsed` windows when `word_timestamps` is off. `single_with_gap>1.0s` counts the `single` windows with more than 1s between the last timestamp and the window end.
- With the fix on, these numbers still describe what the model emitted. They do not measure lost audio.
- Runs in this doc before this change printed the line of every `collapsed` window too and counted them in `full_jump_with_gap`, which overstated the losses.

## Base model vs fine-tuned model

Counts are in the tables above. The fine-tuned model has more `collapsed` windows (4 to 5 against 0 to 1), but `collapsed` windows are recovered with `word_timestamps=True`, so they do not show a difference in lost audio. The `single` windows, the ones that lose audio, are similar: base 4 and fine-tuned 6 with the full prompt, 4 and 4 with the capped prompt. It is one file and one run per model, and the windows of the two runs do not line up, so there is no evidence that fine-tuning makes the loss worse. The base model used here is `ivrit-ai/whisper-large-v3-turbo-ct2`, the fine-tuned model is `portal-whisper-turbu-v3-5000-cosine-ct2`.

**Later finding** (talmud model, `word_timestamps=False`): with `collapsed` windows counted as losses, the fine-tuned model loses audio in 6 windows and the base model in 2. See "Root cause".

The fine-tuned model writes long segments, often one segment of 20 to 28s, while its training data has short segments (next section). The base model writes sentence-level segments. That is a real difference in output style. It matters for subtitles and for how a cut lands, but with `word_timestamps=True` it does not by itself cause lost audio.

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
- The training data is sentence-level with a timestamp pair every few seconds, and it has the cross-over case in most samples. Adding more short sentences is not supported by this data, because it already has short segments.
- **Later finding:** the format does explain the long segments. Every boundary in the labels sits on a pause of at least about 0.3s, because the builder merges segments with smaller gaps. The model learned to split only at pauses, and the lecture has almost none (see "Root cause"). The earlier ideas (too few training steps, speaking style, the prompt) are not needed to explain it; the prompt was tested and ruled out.
- Nothing "adds" timestamps at inference. The model must write them itself. faster-whisper only reads them.
- Training prefixes are shortened to leave room for the target, but faster-whisper did not do that at inference. The shorter prompt fix (above) removes that difference.

## Other settings tested

- `without_timestamps=True`: cuts the audio into hard 30s blocks and loses words at each boundary. The sampled training data has no `<|notimestamps|>` examples. Keep it `False`.
- `condition_on_previous_text=False`: made the problem disappear on a 2-minute crop. The explanation given then (the output gets the full 448 positions) is wrong, since the cap is 224 with any prompt. On the full file with the talmud model it did not remove the problem: 4 `collapsed` windows, mostly 3 timestamps per window.
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
2. CTranslate2 caps each window's output at `min(224, 448 - prompt)` tokens. Dense Hebrew windows are cut off at that limit. Capping the previous text at 128 tokens raises the cap only from 221 to 224.
3. Audio is lost when a window is cut off with no two timestamps in a row to restart from. In the beam-5 run all 6 lossy windows were `collapsed` at 224 tokens. The 612.58s "early close" was probably a cut at the cap too.
4. With `word_timestamps=True`, only `single` windows lose audio. `collapsed` and `open` windows are recovered. The seek fix recovered the 255s junction. The last-window guard is untested.
5. The model writes a boundary only at pauses of about 0.3s or more, because the dataset builder's 0.3s merge removed every other boundary from the labels. In continuous speech this leaves windows with no boundary. This is the root cause; the change is in `create_dataset_segment_merge_change.md`.

## Next steps

- Rerun `33626.mp3` with `--log-windows` and the defaults (seek fix on, 128-token cap). Check that the 612.58s, 255s and 473.8s junctions are recovered and the "תודה רבה" loop at the end is gone.
- Done: the fine-tuned model reproduces the training style on 80 eval samples (5.8 boundaries per window against 6.4 in the labels).
- Optional direct test of the pause dependency: insert about 0.6s of silence at a few sentence ends in the 198.20s window and check whether boundaries appear exactly there.
- Change the merge in the dataset builder (`create_dataset_segment_merge_change.md`), rebuild, retrain, and rerun `33626.mp3` with `--log-windows`.
- `limit_previous_tokens` values matter little now that the cap is known to be 224.
- Port the fix, the prompt limit and the threshold to `engines/faster_whisper_engine.py` and the transcriber service.
- Look into the glued words.
