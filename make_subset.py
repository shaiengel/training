#!/usr/bin/env python3
# coding: utf-8
"""Build small, reproducible train/test subsets of a raw HF dataset.

The source is a HF repo id (downloaded to the HF cache, memory-mapped) or a local save_to_disk DatasetDict
(e.g. dataset_builder's output/dataset_split, already split by source_id); the subset is shuffled by index and
written to --out_dir, and only then pushed to the Hub from disk.
"""

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk


def take_random_subset(source, split, n, seed):
    # load_dataset mis-detects save_to_disk folders, so read local paths with load_from_disk
    if Path(source).exists():
        dataset = load_from_disk(source)[split]
    else:
        dataset = load_dataset(source, split=split)
    return dataset.shuffle(seed=seed).select(range(n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="portal-daf-yomi/portal-daf-yomi-whisper-training", help="HF repo id or local save_to_disk path")
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--test_size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=745)
    parser.add_argument("--out_dir", required=True, help="save_to_disk target")
    parser.add_argument("--push_to_hub", help="HF repo id to push to (private), uploaded from --out_dir")
    args = parser.parse_args()

    subset = DatasetDict(
        {
            "train": take_random_subset(args.source, "train", args.train_size, args.seed),
            "test": take_random_subset(args.source, "test", args.test_size, args.seed),
        }
    )
    print(f"Start saving subset to {args.out_dir}")
    subset.save_to_disk(args.out_dir)
    print(f"End saving subset to {args.out_dir}")

    if args.push_to_hub:
        # Reload from disk so the upload reads the saved files instead of the shuffled source
        load_from_disk(args.out_dir).push_to_hub(args.push_to_hub, private=True)


if __name__ == "__main__":
    main()
