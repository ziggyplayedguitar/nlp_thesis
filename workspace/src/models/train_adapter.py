
import os
from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from adapter_transformers import AutoModelWithHeads
from src.data_tools.czech_data_tools import load_czech_media_data

def main():
    # Load Czech comments
    print("Loading Czech comments...")
    comments_df = load_czech_media_data(data_dir="./data/MediaSource")
    texts = comments_df['text'].dropna().tolist()
    print(f"Loaded {len(texts)} comments.")

    # Initialize model and tokenizer
    model =  AutoModelWithHeads.from_pretrained("../../checkpoints/best_model.pt")  
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

    # Add a new Adapter
    adapter_name = "cz_news_mlm"
    model.add_adapter(adapter_name)
    model.train_adapter(adapter_name)
    print(f"Added adapter: {adapter_name}")

    # Prepare dataset
    dataset = Dataset.from_dict({"text": texts})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./output/czert_adapter_mlm",
        overwrite_output_dir=True,
        num_train_epochs=2,  # 1-3 epochs are usually enough
        per_device_train_batch_size=32,
        save_steps=1000,
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=5e-5,
        report_to="none",
        evaluation_strategy="no"
    )

    # Start training 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    print("Starting Adapter training...")
    trainer.train()

    # === Step 7: Save adapter separately ===
    adapter_dir = Path("./output/czert_adapter_mlm/adapter")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_adapter(str(adapter_dir), adapter_name)
    print(f"Adapter saved to {adapter_dir}")

if __name__ == "__main__":
    main()