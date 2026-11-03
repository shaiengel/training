# Re-splitting the dataset (small eval sets)

The source dataset `portal-daf-yomi/portal-daf-yomi-whisper-training` has a test split of 8,881 rows (74 h, 30 s per row). That is too large to evaluate on during training. Its `source_id` values also all appear in train, so test is a random split by row and shares recordings with train.

`make_resplit.py` pools the source train and test splits and builds three splits that are disjoint by `metadata.source_id`:

| Split | Rows | Hours | Sources | Use |
|---|---|---|---|---|
| train | 108,881 | 907.3 | 1,876 | training |
| validation | 291 | 2.4 | 3 | eval during training |
| test | 736 | 6.1 | 10 | final score only, never used for checkpoint selection |

Whole sources are drawn at random (seed 745) until the target hours are reached, so the sizes land slightly above the 2 h / 6 h targets. Validation has only 3 sources, so its WER is noisy across speakers. For more diversity, change `--seed` or raise `--validation_hours`.

## Commands used

The machine could not reach huggingface.co (SSL certificate error), so the cached copy was used offline.

Inspect the cached dataset: the schema is in `dataset_info.json`, and per-split hours and source overlap were computed from the `metadata` column (script kept outside the repo in `C:\portal\tmp_scripts\stats.py`).

```bash
uv run python C:/portal/tmp_scripts/stats.py
```

Build the new splits (about 5 minutes, writes about 49 GB):

```bash
HF_HUB_OFFLINE=1 uv run python make_resplit.py --out_dir C:/portal/dataset_builder/output/resplit
```

Options: `--validation_hours` (default 2), `--test_hours` (default 6), `--seed` (default 745), `--source`, `--out_dir` (required).

Output: `C:\portal\dataset_builder\output\resplit` (`train/`, `validation/`, `test/`, `dataset_dict.json`). Load it with `datasets.load_from_disk`.

## Why a second processing step is needed

`make_resplit.py` only moves rows between splits. The rows still hold encoded audio and text, and Whisper cannot train on those directly. Training needs 16 kHz log-Mel spectrograms and tokenized transcripts, with timestamp tokens and previous-text context added. `train-whisper.py` computes these in its preprocessing step (`DatasetPreparator` in `preprocess/preperator.py`), which is the slow part.

You can skip `--save_processed` and pass the raw `resplit` straight to `--train_datasets`. Training then redoes that conversion on every start. Preprocessing once with `--save_processed` saves time but is not required.

## Next steps (not run yet)

Upload to S3 (you do this step). Then preprocess the new dataset with the same model as the real run. `--train_datasets` takes Hub ids, so for a local path check how `load_datasets` in `train-whisper.py` resolves it before running:

```bash
uv run python train-whisper.py --output_model_name prep --skip_push_to_hub --model_name ivrit-ai/whisper-large-v3 \
  --train_datasets <train spec> --eval_datasets <validation spec> \
  --save_processed <out> --ds_processor_proc_num 4
```

For the real run, use `--learning_rate 1e-5` from `README-lr-search.md` and keep `--max_eval_set_size` at or below the validation size (291).
