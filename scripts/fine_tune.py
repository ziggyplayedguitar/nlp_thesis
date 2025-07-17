import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Allow running from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "workspace"))

from src.data_tools.dataset import collate_batch
from src.models.predictor import TrollPredictor
from src.models.trainer import TrollDetectorTrainer


class FewShotDataset(Dataset):
    """Dataset for a small set of manually labelled authors."""

    def __init__(self, data, tokenizer_name="distilbert-base-multilingual-cased", max_length=96, comments_per_user=10):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.comments_per_user = comments_per_user

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        comments = list(item["text"])

        if len(comments) > self.comments_per_user:
            comments = comments[: self.comments_per_user]
        elif len(comments) < self.comments_per_user and len(comments) > 0:
            comments = comments + [comments[-1]] * (self.comments_per_user - len(comments))

        encodings = self.tokenizer(
            comments,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "label": torch.tensor(item["label"], dtype=torch.float),
            "author": item["author"],
        }


def create_fewshot_dataset(annotations_df: pd.DataFrame, comments_df: pd.DataFrame):
    fewshot_data = []
    for _, row in annotations_df.iterrows():
        if row.get("label") == -1:
            continue
        author_comments = comments_df[comments_df["author"] == row["author"]]
        fewshot_data.append(
            {
                "author": row["author"],
                "text": author_comments["text"].tolist(),
                "label": row["label"],
            }
        )
    return fewshot_data


def load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main(args):
    annotations_df = load_dataframe(args.annotations)
    comments_df = load_dataframe(args.comments)

    fewshot_data = create_fewshot_dataset(annotations_df, comments_df)
    dataset = FewShotDataset(
        fewshot_data,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        comments_per_user=args.comments_per_user,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    predictor = TrollPredictor(
        model_path=args.checkpoint,
        comments_per_user=args.comments_per_user,
        max_length=args.max_length,
    )
    model = predictor.model

    trainer = TrollDetectorTrainer(
        model=model,
        train_loader=dataloader,
        val_loader=dataloader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.output_dir,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a TrollDetector checkpoint on annotated comments")
    parser.add_argument("--annotations", required=True, help="CSV file with author labels")
    parser.add_argument("--comments", required=True, help="Parquet/CSV file with comments including author and text")
    parser.add_argument("--checkpoint", required=True, help="Path to base checkpoint")
    parser.add_argument("--output_dir", default="./checkpoints/fine_tuned", help="Directory to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=7, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--comments_per_user", type=int, default=10, help="Number of comments per author")
    parser.add_argument("--max_length", type=int, default=96, help="Maximum sequence length")
    parser.add_argument("--tokenizer", default="distilbert-base-multilingual-cased", help="Tokenizer name")
    args = parser.parse_args()
    main(args)
