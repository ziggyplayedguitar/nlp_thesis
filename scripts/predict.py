import argparse
from pathlib import Path
import sys
import pandas as pd

# Add project src directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[1] / "workspace"
sys.path.append(str(PROJECT_ROOT))

from src.models.predictor import TrollPredictor


def load_input_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input file type: {path.suffix}")

    if "author" not in df.columns or "text" not in df.columns:
        raise ValueError("Input data must contain 'author' and 'text' columns")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run troll predictions on a dataset")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("input_file", help="CSV or Parquet file with comments")
    parser.add_argument("output_file", help="Where to save CSV predictions")
    parser.add_argument("--comments_per_user", type=int, default=10, help="Number of comments per user")
    parser.add_argument("--max_length", type=int, default=96, help="Tokenizer max length")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--device", default=None, help="Force device (cpu or cuda)")
    parser.add_argument("--adapter_path", default=None, help="Optional adapter weights path")
    parser.add_argument("--use_adapter", action="store_true", help="Enable adapter layers")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    df = load_input_file(input_path)

    predictor = TrollPredictor(
        model_path=args.checkpoint,
        device=args.device,
        comments_per_user=args.comments_per_user,
        max_length=args.max_length,
        threshold=args.threshold,
        adapter_path=args.adapter_path,
        use_adapter=args.use_adapter,
    )

    predictions = predictor.predict_authors(df)
    predictions.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
