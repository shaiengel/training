#!/usr/bin/env python3
# coding: utf-8
"""Re-split a raw HF dataset into train / validation / test with small, source-disjoint eval sets.

Train and test of the source are pooled, whole sources (metadata.source_id) are drawn at random into
validation and test until their row targets are met, and everything else becomes train. Splitting by
source keeps segments of the same recording from landing on both sides. The result is saved locally.
"""

import argparse
import random

from datasets import DatasetDict, concatenate_datasets, load_dataset


def pick_sources(sources_rows, target_rows, taken):
    """Add shuffled sources to the pick until target_rows is reached; skip sources that would overshoot by >25%."""
    picked, rows = [], 0
    for source, n in sources_rows:
        if source in taken or rows >= target_rows:
            continue
        if rows + n > target_rows * 1.25:
            continue
        picked.append(source)
        rows += n
    return picked, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="portal-daf-yomi/portal-daf-yomi-whisper-training")
    parser.add_argument("--validation_hours", type=float, default=2.0, help="eval set used during training")
    parser.add_argument("--test_hours", type=float, default=6.0, help="final held-out score, never used for selection")
    parser.add_argument("--seed", type=int, default=745)
    parser.add_argument("--out_dir", required=True, help="save_to_disk target")
    args = parser.parse_args()

    pooled = concatenate_datasets([load_dataset(args.source, split=s) for s in ("train", "test")])
    meta = pooled.select_columns(["metadata"])["metadata"]
    source_ids = [m["source_id"] for m in meta]
    durations = [m["duration"] for m in meta]

    rows_by_source = {}
    for idx, source in enumerate(source_ids):
        rows_by_source.setdefault(source, []).append(idx)

    rng = random.Random(args.seed)
    order = sorted(rows_by_source)
    rng.shuffle(order)
    sources_rows = [(s, len(rows_by_source[s])) for s in order]

    # Rows are fixed-length (30 s) but convert through the real mean duration to stay honest
    mean_dur = sum(durations) / len(durations)
    val_target = round(args.validation_hours * 3600 / mean_dur)
    test_target = round(args.test_hours * 3600 / mean_dur)

    test_sources, _ = pick_sources(sources_rows, test_target, set())
    val_sources, _ = pick_sources(sources_rows, val_target, set(test_sources))
    held_out = set(test_sources) | set(val_sources)

    def indices(sources):
        return sorted(i for s in sources for i in rows_by_source[s])

    train_idx = sorted(i for s, rows in rows_by_source.items() if s not in held_out for i in rows)
    split_idx = {"train": train_idx, "validation": indices(val_sources), "test": indices(test_sources)}

    for name, idx in split_idx.items():
        hours = sum(durations[i] for i in idx) / 3600
        n_sources = len({source_ids[i] for i in idx})
        print(f"{name}: {len(idx)} rows, {hours:.2f} h, {n_sources} sources")

    result = DatasetDict({name: pooled.select(idx) for name, idx in split_idx.items()})
    print(f"Start saving to {args.out_dir}")
    result.save_to_disk(args.out_dir)
    print(f"End saving to {args.out_dir}")


if __name__ == "__main__":
    main()
