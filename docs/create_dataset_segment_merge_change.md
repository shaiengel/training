# Dataset builder: merge segments at pauses or sentence ends, not only at pauses

The change is needed in two places that contain the same code:

| | file | merge function | called at |
|---|---|---|---|
| **dataset builder** (built `dataset_processed_large_turbo`) | `C:\portal\dataset_builder\src\dataset_builder\infrastructure\whisper_dataset_generator.py` | `WhisperDatasetGenerator._merge_slice_segments`, line 286 | `prepare_training_dataset`, line 532 |
| training repo | `C:\portal\training\create_dataset.py` | `merge_slice_segments`, line 301 | `prepare_training_dataset`, line 608 |

Line numbers below refer to the dataset builder unless marked otherwise.

## Why

The fine-tuned model (`whisper-ivrit-talmud-large-v3-turbo`) writes a segment boundary only where the speaker pauses for about 0.3s or more. In lessons with continuous speech it writes about one boundary per 30s window, sometimes none. A window with no boundary reaches CTranslate2's 224-token output cap with no timestamp to restart from, and faster-whisper skips the rest of the window. That is the text lost in `33626.mp3` (see `faster_whisper_missing_text_investigation.md`).

The model learned this from the training labels:

- Of 36,557 boundaries in `dataset_processed_large_turbo`, none has a gap of 0.2s or less between one segment's end and the next one's start (median 0.62s).
- Every boundary the model writes on `33626.mp3` has a gap of 0.28s or more.
- On the dataset's own eval examples the model writes 5.8 boundaries per window (labels: 6.4), so timestamp prediction itself works. The base model `ivrit-ai/whisper-large-v3-turbo` writes 5 to 9 boundaries per window on the same lesson.

## The cause in the code

`_merge_slice_segments` runs on every slice after slicing (line 532, default arguments). It merges two neighbouring segments when the gap between them is under 0.3s:

```python
gap = current_segment["start"] - prev_segment["end"]
if gap < merge_below_gap_threshold:   # 0.3
    prev_segment["end"] = current_segment["end"]
    prev_segment["text"] = prev_segment["text"] + current_segment["text"]
    prev_segment["word_scores"] = prev_segment["word_scores"] + current_segment["word_scores"]
```

`create_dataset.py` has the same rule (without the `word_scores` line).

## Where the segments come from in the builder

The builder does not use the stable-ts segments directly:

1. `JsonParser` (`json_parser.py`) flattens the stable-ts JSON into a list of words with their times.
2. `SegmentParser` (`segment_parser.py`) matches those words one by one to the words of the lesson's VTT file and puts each word into the segment of its VTT cue (line 103). Each VTT cue becomes one `Segment`, with the start of its first word and the end of its last word.
3. `_generate_slices` (line 130) cuts these segments into 30s slices, then `_merge_slice_segments` merges them.

So the input to the merge is one segment per VTT cue. Checked on the seven lessons below against `portal-daf-yomi/portal-daf-yomi-whisper-training` (built by the same builder): every segment boundary in the labels falls on a stable-ts segment boundary (100%), and 99% to 100% of the stable-ts boundaries with a gap of 0.3s or more appear in the labels. The VTT cues are the stable-ts segments, so the measurements on the stable-ts JSON apply to the builder.

## What the input looks like

Measured on seven stable-ts transcripts of lessons (`102904`, `152156`, `116961`, `109717`, `189988`, `185650`, `153768`; 24 to 72 minutes each). All seven have the same `regroup_history`: stable-ts splits at `.` `?`, commas, gaps of 0.5s or more, and 70 characters.

| lesson | input segment median | under 2 words | gap median | gaps under 0.3s | sentence-end boundaries removed by the current merge |
|---|---|---|---|---|---|
| 102904 | 1.1s, 4 words | 7.8% | 0.18s | 67% | 59% |
| 152156 | 2.3s, 5 words | 4.1% | 0.36s | 44% | 29% |
| 116961 | 1.5s, 4 words | 7.9% | 0.44s | 40% | 34% |
| 109717 | 1.8s, 5 words | 9.2% | 0.58s | 28% | 21% |
| 189988 | 1.8s, 6 words | 3.2% | 0.38s | 39% | 24% |
| 185650 | 2.4s, 6 words | 4.9% | 0.20s | 61% | 53% |
| 153768 | 1.4s, 4 words | 12.3% | 0.24s | 56% | 51% |

Almost every input segment ends in `,` `.` or `?`. The current merge removes 21% to 59% of the real sentence ends, depending on how fast the speaker talks. It also creates long segments: in 185650 (a fast speaker) up to 73.6s, with 18% of segments over 10s, so slicing turns many of its samples into one segment for the whole window.

Boundaries per 30s of audio for different rules (computed on the whole lesson, before slicing):

| rule | 102904 | 152156 | 116961 | 109717 | 189988 | 185650 | 153768 | notes |
|---|---|---|---|---|---|---|---|---|
| gap < 0.3s (current) | 6.1 | 5.3 | 7.6 | 7.4 | 7.4 | 4.0 | 6.3 | boundaries only at pauses, segments up to 73.6s |
| no merging | 18.4 | 9.6 | 12.7 | 10.3 | 12.1 | 10.2 | 14.1 | mostly at commas, about 37 timestamp tokens per window in 102904 |
| punctuation only, no length limit | 7.3 | 3.9 | 5.6 | 4.1 | 5.8 | 4.3 | 7.3 | segments up to 534.6s (109717) |
| punctuation only, 10s limit | 7.8 | 5.0 | 6.3 | 5.7 | 6.5 | 5.4 | 7.7 | fewer boundaries than today in slow lessons: comma boundaries at real pauses are lost |
| **sentence end or pause, 10s limit (proposed)** | **10.8** | **6.7** | **9.8** | **8.7** | **8.9** | **6.9** | **10.6** | more boundaries than today in every lesson, segments at most 10s |

With the proposed rule, 14% to 43% of the boundaries have a gap under 0.3s (in the current dataset none do), and no segment is longer than 10s. An 8s limit gives almost the same results, so 10s is enough.

The real labels have fewer boundaries per sample than these per-30s numbers (today 3.4 to 6.5 for the same lessons), because a crossover segment at the end of a sample is not counted and the next sample starts again at the last complete segment.

## The change

Replace the gap rule in `_merge_slice_segments` with these rules. Process segments in order and decide for each segment whether to merge it into the previous one.

1. **Keep the boundary after a sentence end.** If the previous segment ends with `.` `?` or `!`, keep the boundary, whatever the gap. This adds the boundaries in continuous speech that the current rule removes.
2. **Keep the boundary at a pause.** If the gap to the previous segment is at least `pause_threshold` seconds (default 0.3), keep the boundary, whatever the punctuation. These are the boundaries the current rule already keeps. Without this rule, slow lessons get fewer boundaries than today (see the table).
3. **Merge across the remaining boundaries (commas or no punctuation, no pause) while the merged segment stays short.** Merge only if the merged segment would be at most `max_merged_duration` seconds long (default 10). Otherwise keep the boundary. Without this limit, a stretch with no sentence punctuation becomes one segment (534.6s in 109717), which brings the original problem back.
There is no rule that merges short segments by word count. Lessons have real one- and two-word segments after a pause or at a sentence end ("בכלל.", "תן דוגמה,"), and the current dataset keeps them. Always merging segments under 2 words would remove those boundaries too and teach the model that short segments do not exist. With the rules above, a short segment survives only at a sentence end or a pause; a short fragment at a comma with no pause is merged by rule 3. In the seven lessons, 5.5% to 21.4% of the resulting segments have one or two words.

The difference from the current code: today a boundary needs a pause. With these rules a boundary needs a pause **or** a sentence end, so the model learns to split continuous speech at sentence ends.

When merging, keep concatenating `word_scores` as the builder does today (line 344).

### Sentence-end check

The merge sees the raw segment text, built from the stable-ts words. `normalize_asr_text` runs only later, when the label is written (line 412), so the raw text can still contain typographic quotes. Before checking the last character, strip trailing whitespace and closing quotes or brackets: `"` `'` `”` `’` `״` `)`.

### Keep the crossover behavior

The last segment of a slice can be a crossover segment: it has only `start`, no `end` or `text`, because it continues in the next slice. The current code skips it (line 331: it only merges segments that have `start`, `end` and `text`). Keep that:

- Apply rules 1 to 3 only to segments that have `start`, `end` and `text`. Never merge anything into the crossover segment: that would move the crossover start earlier and drop text from the slice.

`_generate_slices` decides where the next slice starts from the input segments (`next_slice_start = input_segments[curr_input_segment_idx - 1].end`, line 240), before merging. Merging never changes the end of the last complete segment, so slice boundaries do not change.

### Parameters

Replace `merge_below_gap_threshold` with:

| parameter | default | meaning |
|---|---|---|
| `pause_threshold` | 0.3 | a gap of at least this many seconds always keeps the boundary (rule 2) |
| `max_merged_duration` | 10.0 | longest segment that rule 3 may create, in seconds |

In the builder there are no command-line arguments. `LessonProcessor` (`services/processor.py:91`) calls `prepare_training_dataset` with fixed values (`slice_length=30`, thresholds 0), and settings come from `Config` (`config.py`, environment variables with defaults). Add the two parameters to `prepare_training_dataset`, pass them to `_merge_slice_segments` at line 532, and pass them from `processor.py:91`. If they should be configurable, add them to `Config` the same way as the other fields.

In `create_dataset.py`, add them as command-line arguments next to the existing ones (lines 688-771) and pass them at line 608.

## Compatibility with long-form transcription

The dataset follows [Whisper Long-Form Transcription](https://medium.com/@yoad/whisper-long-form-transcription-1924c94a9b86), which says training data must match how long-form inference works. The change keeps every requirement the article lists:

| Article requirement | Today | With the change |
|---|---|---|
| Always use timestamps (long-form needs them) | yes | unchanged |
| Timestamps come in start/end pairs and always increase | yes | unchanged. Merging only removes boundaries, so order and pairing stay valid. |
| Speech that runs past 30s leaves its segment open: an opening timestamp with no text | yes (`_generate_slices`) | unchanged. Nothing is merged into or out of the crossover segment. |
| The next window starts at the open segment | yes (`next_slice_start = ... .end`) | unchanged. Merging happens after slicing and never moves the end of the last complete segment. |
| Previous text after `<|startofprev|>` | yes | unchanged |

The article does not mention merging segments or a gap threshold. Its own examples have touching timestamps at every boundary, for example `...Here comes a pause.<|6.00|><|6.00|>` and `...to start with.<|21.72|><|21.72|>`: boundaries with no gap, which the current 0.3s merge removes from the dataset completely. So the change brings the labels closer to the article's format, not further.

The 0.3s merge comes from ivrit.ai's `create_dataset.py` (upstream commit `c3f47d0`, "More reliable training dataset creation"), which the builder copied. The article mentions a companion post about training Whisper Turbo at ivrit.ai that might explain it. That post was not found (it is not on the author's Medium page), so it has not been checked whether it gives a reason for the 0.3s rule that this change would break.

## Tests

- **Dataset builder:** there are no tests for the merge. `tests/test_transcript_spacing.py` is the only test file. Add a new test file under `tests/` for `_merge_slice_segments`.
- **Training repo:** `tests/create_dataset/test_merge_slice_segments.py` tests the gap rule (for example `merge_below_gap_threshold=1.0` at line 165) and needs updating.

Cases to cover in both:

- boundary after `.` `?` `!` kept even with a gap of 0
- boundary after a sentence end followed by a closing quote kept (`...הן."`, `...הן.”`)
- boundary after a comma kept when the gap is 0.3s or more
- segments joined across a comma with a gap under 0.3s when the result is at most 10s
- boundary at a comma with a gap under 0.3s kept when merging would exceed 10s
- a one-word segment after a pause or a sentence end kept (`...הן.<|25.62|><|26.14|> בכלל.<|27.28|>`)
- a one-word segment at a comma with no pause merged into the previous one
- `word_scores` concatenated when segments are merged (builder)
- crossover segment (no `end`/`text`) never merged
- empty slice and single-segment slice unchanged

## Checking the result

Before retraining, rebuild the dataset for a few lessons (for example the seven above) and check the labels:

- boundaries per sample: more than today for every lesson (today 3.4 to 6.5 per sample for the seven lessons)
- gaps between segment end and next start: 14% to 43% under 0.3s, as in the seven lessons (today: none under 0.2s)
- longest text run without a timestamp: should not grow (today: median 48 tokens, 0.5% at 200 or more)
- share of targets over 224 tokens (today 14.8%): boundaries add 2 tokens each, so this may rise slightly

After retraining, run `run_whisper.py --log-windows` on `33626.mp3`. Windows should have several boundaries even in continuous speech, and there should be no `collapsed` windows.
