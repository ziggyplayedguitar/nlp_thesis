import argparse
from pathlib import Path
import sys

import pandas as pd
from torch.utils.data import DataLoader

# Ensure src package is importable when running as a script
ROOT_DIR = Path(__file__).resolve().parents[1] / "workspace"
sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TrollDetector model")
    parser.add_argument("--train-data", type=str, required=True,
                        help="Path to the training parquet file")
    parser.add_argument("--val-data", type=str, required=True,
                        help="Path to the validation parquet file")
    parser.add_argument("--test-data", type=str, default=None,
                        help="Optional path to the test parquet file")

    # Model and tokenizer parameters
    parser.add_argument("--model-name", type=str,
                        default="distilbert-base-multilingual-cased",
                        help="Hugging Face model name")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to a pre-trained adapter")
    parser.add_argument("--use-enhanced-attention", action="store_true",
                        help="Use enhanced tweet attention mechanism")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--dropout-rate", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Number of warmup steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping value")

    # Dataset parameters
    parser.add_argument("--max-length", type=int, default=96,
                        help="Maximum sequence length")
    parser.add_argument("--comments-per-user", type=int, default=10,
                        help="Number of comments per user")

    # Misc
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory to store checkpoints")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Log metrics to Weights & Biases")

    return parser.parse_args()


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "author" not in df.columns or "text" not in df.columns:
        raise ValueError("Parquet file must contain 'author' and 'text' columns")
    return df


def build_dataset(df: pd.DataFrame, args: argparse.Namespace):
    from src.data_tools.dataset import TrollDataset

    return TrollDataset(
        df,
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        comments_per_user=args.comments_per_user,
        label_column="troll",
        normalize_labels=True,
    )


def main() -> None:
    args = parse_args()

    from src.data_tools.dataset import collate_batch
    from src.models.bert_model import TrollDetector
    from src.models.trainer import TrollDetectorTrainer

    # Load data
    train_df = load_dataframe(args.train_data)
    val_df = load_dataframe(args.val_data)
    test_df = load_dataframe(args.test_data) if args.test_data else None

    # Create datasets
    train_dataset = build_dataset(train_df, args)
    val_dataset = build_dataset(val_df, args)
    test_dataset = build_dataset(test_df, args) if test_df is not None else None

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                   collate_fn=collate_batch)
        if test_dataset is not None else None
    )

    # Model and trainer
    model = TrollDetector(
        model_name=args.model_name,
        dropout_rate=args.dropout_rate,
        adapter_path=args.adapter_path,
        use_enhanced_attention=args.use_enhanced_attention,
    )
    trainer = TrollDetectorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.output_dir,
        use_wandb=args.use_wandb,
    )

    trainer.train()
    best_path = Path(args.output_dir) / "best_model.pt"
    print(f"Best checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()
