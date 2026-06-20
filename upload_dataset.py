from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from clearml import Dataset
from datasets import load_dataset

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a sentiment dataset to ClearML Dataset."
    )
    parser.add_argument("--project-name", type=str, default="itmo-mlops")
    parser.add_argument("--dataset-name", type=str, default="tweet_eval_sentiment")
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/tweet_eval.csv"),
    )
    return parser.parse_args()


def build_frame(sample_size: int, seed: int) -> pd.DataFrame:
    """Download tweet_eval sentiment split and map numeric labels to names."""
    
    raw = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train")
    df = pd.DataFrame({"text": raw["text"], "label_id": raw["label"]})
    df["label"] = df["label_id"].map(LABEL_NAMES)

    if 0 < sample_size < len(df):
        df = df.sample(sample_size, random_state=seed).reset_index(drop=True)
    return df


def main() -> None:
    """Save the dataset locally and publish a finalized ClearML Dataset version."""
    args = parse_args()

    df = build_frame(sample_size=args.sample_size, seed=args.seed)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved rows: {len(df)} to {args.output_csv}")

    dataset = Dataset.create(
        dataset_name=args.dataset_name, dataset_project=args.project_name
    )
    dataset.add_files(path=str(args.output_csv))
    dataset.finalize(auto_upload=True)

    print(f"Dataset ID: {dataset.id}")


if __name__ == "__main__":
    main()
