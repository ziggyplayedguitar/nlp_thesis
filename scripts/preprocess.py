"""Command line interface for preprocessing datasets."""

import argparse
import sys
from pathlib import Path

# Make src modules importable when running from repository root
sys.path.append(str(Path(__file__).resolve().parents[1] / "workspace"))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess raw data and generate train/val/test splits",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing raw datasets",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save processed parquet files",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="Proportion of authors in the training split",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Proportion of authors in the validation split",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Proportion of authors in the test split",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    parser.add_argument(
        "--max-tweets-per-source",
        type=int,
        default=None,
        help="Limit tweets loaded per source dataset",
    )
    parser.add_argument(
        "--max-tweets-per-author",
        type=int,
        default=None,
        help="Limit tweets kept per author",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional prefix for output filenames",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Heavy imports happen after argument parsing so `--help` works without
    # requiring optional dependencies like PyTorch.
    from src.data_tools.preprocessor import load_and_clean_data
    from src.data_tools.dataset import create_data_splits

    df = load_and_clean_data(
        args.input_dir,
        max_tweets_per_source=args.max_tweets_per_source,
        max_tweets_per_author=args.max_tweets_per_author,
    )

    df = df.rename(columns={"account": "author", "tweet": "text"})

    train_df, val_df, test_df = create_data_splits(
        df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_seed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{args.prefix}_" if args.prefix else ""
    for name, split_df in {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }.items():
        path = out_dir / f"{prefix}{name}.parquet"
        split_df.to_parquet(path, index=False)
        print(f"Saved {name} split to {path}")


if __name__ == "__main__":
    main()

