import pandas as pd
import numpy as np
import json
from pathlib import Path
import re
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
import matplotlib.pyplot as pl
import uuid
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TweetPreprocessor:
    def __init__(self):
        self.url_pattern = re.compile(r'https?://\S+\b')
        self.pic_pattern = re.compile(r'pic\.twitter\.com/\w+\b')
        self.multiple_whitespace = re.compile(r'\s+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.mention_pattern = re.compile(r'@\w+')
        self.quote_pattern = re.compile(r'"([^"]*)"')
        
        # Define custom stop words for social media content
        self.custom_stop_words = {
            't', 's', 'm', 'rt', 'http', 'https',  # Common Twitter artifacts
            'amp', 'co', 've', 'll', 'd', 're',    # Contractions and common artifacts
            'twitter', 'tweet', 'retweet'         # Twitter-specific terms
        }

    def preprocess_tweet(self, text: str) -> str:
        """Basic tweet preprocessing while keeping hashtags and mentions"""
        if pd.isna(text):
            return ""
        
        # Remove URLs, Twitter pics, hashtags, and mentions
        # text = str(text).lower()
        text = self.url_pattern.sub('', text) 
        text = self.pic_pattern.sub('', text)
        text = self.hashtag_pattern.sub('', text)
        text = self.mention_pattern.sub('', text)
        text = self.quote_pattern.sub('', text)

        # Split into words and filter out stop words
        words = text.split()
        words = [word for word in words if word not in self.custom_stop_words]

        # Rejoin and normalize whitespace
        text = ' '.join(words)
        text = self.multiple_whitespace.sub(' ', text)

        return text.strip()

def load_twitter_json(json_folder: Path) -> pd.DataFrame:
    """Load and preprocess tweets from all JSON files in the specified folder."""
    logger.info("Loading Twitter JSON data from non_troll_politics folder...")
    all_tweets = []
    
    json_files = list(json_folder.glob("*.json"))
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                tweet_text = entry.get("full_text", "")
                account_name = entry["user"].get("screen_name", "")
                if tweet_text and account_name:
                    all_tweets.append({"account": account_name, "tweet": tweet_text, "troll": 0})
    
    return pd.DataFrame(all_tweets)

# TODO Remove UNK tokens
# TODO Train only on english data maybe
def load_and_clean_data(data_dir: str) -> pd.DataFrame:
    """Load and combine all datasets including Twitter JSON files."""
    data_path = Path(data_dir)
    preprocessor = TweetPreprocessor()
    
    # Load Russian troll tweets
    logger.info("Loading Russian troll tweets...")
    troll_files = list(data_path.glob("russian_troll_tweets/*.csv"))
    troll_tweets = pd.concat([pd.read_csv(f) for f in troll_files])
    troll_tweets = troll_tweets[['author', 'content', 'language']]
    troll_tweets.rename(columns={'author': 'account', 'content': 'tweet'}, inplace=True)
    troll_tweets['troll'] = 1

    # Load Sentiment140 tweets
    logger.info("Loading Sentiment140 tweets...")
    sentiment_path = data_path / "sentiment_tweets/training.1600000.processed.noemoticon.csv"
    sentiment_tweets = pd.read_csv(sentiment_path, encoding='Latin-1',
                                 names=['target', 'id', 'date', 'flag', 'username', 'tweet'])
    sentiment_tweets = sentiment_tweets[['username', 'tweet']]
    sentiment_tweets.rename(columns={'username': 'account'}, inplace=True)
    sentiment_tweets['troll'] = 0
    sentiment_tweets['language'] = 'en'

    # Load celebrity tweets
    logger.info("Loading celebrity tweets...")
    celeb_files = list(data_path.glob("celebrity_tweets/*.csv"))
    celeb_tweets = pd.concat([pd.read_csv(f) for f in celeb_files])
    if 'text' in celeb_tweets.columns:
        celeb_tweets.rename(columns={'text': 'tweet'}, inplace=True)
    if 'author' in celeb_tweets.columns:
        celeb_tweets.rename(columns={'author': 'account'}, inplace=True)
    celeb_tweets = celeb_tweets[['account', 'tweet']]
    celeb_tweets['troll'] = 0
    celeb_tweets['language'] = 'en'
    
    # Load Twitter JSON files from "non_troll_politics" folder
    logger.info("Loading manualy scraped tweets...")
    twitter_data = pd.DataFrame()
    json_folder = data_path / "non_troll_politics"
    if json_folder.exists():
        twitter_data = load_twitter_json(json_folder)
        twitter_data['language'] = 'en'
    
    # Load Parquet files from "information_operations" folder
    logger.info("Loading information operations tweets...")
    parquet_folder = data_path / "information_operations"
    parquet_files = list(parquet_folder.glob("*.parquet"))
    parquet_data = pd.concat([pd.read_parquet(f) for f in parquet_files])
    parquet_data = parquet_data[['accountid', 'post_text', 'is_control', 'post_language']]
    parquet_data.rename(columns={
        'accountid': 'account',
        'post_text': 'tweet',
        'post_language': 'language'
    }, inplace=True)
    parquet_data['troll'] = ~parquet_data['is_control']  # troll is True when is_control is False
    parquet_data = parquet_data.drop('is_control', axis=1)

    # Load files from machova et al
    logger.info("Loading data collected by Machova...")
    nontroll_path = data_path / "machova/Is_not_troll_body.csv"
    nontroll_df = pd.read_csv(nontroll_path)
    nontroll_df['troll'] = 0
    troll_path = data_path / "machova/Is_troll_body.csv"
    troll_df = pd.read_csv(troll_path)
    troll_df['troll'] = 1
    # Add account column and rename existing body column
    combined_df = pd.concat([troll_df, nontroll_df], ignore_index=True)
    combined_df['account'] = [f"user_{i}" for i in range(len(combined_df))]
    combined_df = combined_df.rename(columns={
        'body': 'tweet'
    })
    combined_df['language'] = 'cs'

    # Combine all datasets
    logger.info("Combining datasets...")
    all_tweets = pd.concat([
        troll_tweets[['account', 'tweet', 'troll', 'language']],
        sentiment_tweets[['account', 'tweet', 'troll', 'language']],
        celeb_tweets[['account', 'tweet', 'troll', 'language']],
        twitter_data[['account', 'tweet', 'troll', 'language']],
        parquet_data[['account', 'tweet', 'troll', 'language']],
        combined_df[['account', 'tweet', 'troll', 'language']]
    ], ignore_index=True)

    # Apply preprocessing to tweet text
    all_tweets['tweet'] = all_tweets['tweet'].apply(preprocessor.preprocess_tweet)

    # Remove empty tweets after preprocessing
    all_tweets = all_tweets[all_tweets['tweet'].str.len() > 0]
    
    # Remove accounts with very few tweets
    logger.info("Filtering accounts with few tweets...")
    account_counts = all_tweets.groupby('account').size()
    valid_accounts = account_counts[account_counts >= 10].index
    all_tweets = all_tweets[all_tweets['account'].isin(valid_accounts)]
    
    return all_tweets

class TrollTweetDataset(Dataset):
    def __init__(self, 
                 tweets_df: pd.DataFrame,
                 tokenizer_name: str = "distilbert-base-multilingual-cased",
                 max_length: int = 128,
                 tweets_per_account: int = 10):
        self.tweets_df = tweets_df
        self.preprocessor = TweetPreprocessor()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.tweets_per_account = tweets_per_account
        
        # Group tweets by account
        self.accounts = list(tweets_df.groupby('account'))
        
    def __len__(self) -> int:
        return len(self.accounts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        account_name, account_tweets = self.accounts[idx]
        
        # Sample tweets for this account
        if len(account_tweets) > self.tweets_per_account:
            account_tweets = account_tweets.sample(n=self.tweets_per_account, random_state=42)
        elif len(account_tweets) < self.tweets_per_account:
            # If we have fewer tweets than needed, sample with replacement
            account_tweets = account_tweets.sample(n=self.tweets_per_account, replace=True, random_state=42)
        
        # Preprocess tweets
        tweets = [self.preprocessor.preprocess_tweet(t) for t in account_tweets['tweet']]
        
        # Tokenize all tweets with fixed padding
        encodings = self.tokenizer(
            tweets,
            padding='max_length',  # Changed to max_length
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'label': torch.tensor(account_tweets['troll'].iloc[0], dtype=torch.long)
        }

def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function to handle batches of tweet sequences.
    
    Args:
        batch: List of dictionaries containing input_ids, attention_mask, and label tensors
        
    Returns:
        Dictionary containing batched tensors for input_ids, attention_mask, and labels
    """
    # Stack all tensors from the batch
    input_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
    attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0)
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels
    }

def save_data_to_json(data_dir: str, output_dir: str, preprocess: bool = False) -> None:
    """
    Load and save each dataset separately to individual JSON files.
    
    Args:
        data_dir (str): Directory containing all the raw data files
        output_dir (str): Directory where the JSON files will be saved
        preprocess (bool): Whether to preprocess the text data before saving
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    preprocessor = TweetPreprocessor() if preprocess else None
    
    def process_text(text: str) -> str:
        """Helper function to optionally preprocess text"""
        if preprocess and preprocessor:
            return preprocessor.preprocess_tweet(text)
        return text
    
    # Save Russian troll tweets
    logger.info("Saving Russian troll tweets...")
    troll_files = list(data_path.glob("russian_troll_tweets/*.csv"))
    troll_comments = []
    for f in troll_files:
        troll_tweets = pd.read_csv(f)
        for _, row in troll_tweets.iterrows():
            troll_comments.append({
                "docId": str(uuid.uuid4()),
                "docType": "Comment",
                "author": row["author"],
                "content": process_text(row["content"]),
                "troll": 1,
                "language": row.get("language", "unknown")
            })
    
    with open(output_path / "russian_trolls.json", "w", encoding="utf-8") as f:
        json.dump({"comments": troll_comments}, f, ensure_ascii=False, indent=4)
    
    # Save Sentiment140 tweets
    logger.info("Saving Sentiment140 tweets...")
    sentiment_path = data_path / "sentiment_tweets/training.1600000.processed.noemoticon.csv"
    sentiment_comments = []
    sentiment_tweets = pd.read_csv(sentiment_path, encoding='Latin-1',
                                 names=['target', 'id', 'date', 'flag', 'username', 'tweet'])
    for _, row in sentiment_tweets.iterrows():
        sentiment_comments.append({
            "docId": str(uuid.uuid4()),
            "docType": "Comment",
            "author": row["username"],
            "content": process_text(row["tweet"]),
            "troll": 0,
            "language": "en"
        })
    
    with open(output_path / "sentiment140.json", "w", encoding="utf-8") as f:
        json.dump({"comments": sentiment_comments}, f, ensure_ascii=False, indent=4)
    
    # Save celebrity tweets
    logger.info("Saving celebrity tweets...")
    celeb_files = list(data_path.glob("celebrity_tweets/*.csv"))
    celeb_comments = []
    for f in celeb_files:
        celeb_tweets = pd.read_csv(f)
        for _, row in celeb_tweets.iterrows():
            celeb_comments.append({
                "docId": str(uuid.uuid4()),
                "docType": "Comment",
                "author": row.get("author", row.get("username", "unknown")),
                "content": process_text(row.get("text", row.get("tweet", ""))),
                "troll": 0,
                "language": "en"
            })
    
    with open(output_path / "celebrity_tweets.json", "w", encoding="utf-8") as f:
        json.dump({"comments": celeb_comments}, f, ensure_ascii=False, indent=4)
    
    # Save Twitter JSON files
    logger.info("Saving manually scraped tweets...")
    json_folder = data_path / "non_troll_politics"
    if json_folder.exists():
        json_comments = []
        for json_file in json_folder.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    json_comments.append({
                        "docId": str(uuid.uuid4()),
                        "docType": "Comment",
                        "author": entry["user"].get("screen_name", ""),
                        "content": process_text(entry.get("full_text", "")),
                        "troll": 0
                    })
        
        with open(output_path / "manual_tweets.json", "w", encoding="utf-8") as f:
            json.dump({"comments": json_comments}, f, ensure_ascii=False, indent=4)
    
    # Save information operations tweets
    logger.info("Saving information operations tweets...")
    parquet_folder = data_path / "information_operations"
    parquet_comments = []
    for parquet_file in parquet_folder.glob("*.parquet"):
        parquet_data = pd.read_parquet(parquet_file)
        for _, row in parquet_data.iterrows():
            parquet_comments.append({
                "docId": str(uuid.uuid4()),
                "docType": "Comment",
                "author": row["accountid"],
                "content": process_text(row["post_text"]),
                "troll": 0 if row.get("is_control", False) else 1,
                "language": row.get("post_language", "unknown")
            })
    
    with open(output_path / "information_operations.json", "w", encoding="utf-8") as f:
        json.dump({"comments": parquet_comments}, f, ensure_ascii=False, indent=4)
    
    # Save Machova et al data
    logger.info("Saving Machova et al data...")
    machova_comments = []
    
    # Load and save non-troll data
    nontroll_path = data_path / "machova/Is_not_troll_body.csv"
    if nontroll_path.exists():
        nontroll_df = pd.read_csv(nontroll_path)
        for _, row in nontroll_df.iterrows():
            machova_comments.append({
                "docId": str(uuid.uuid4()),
                "docType": "Comment",
                "author": f"user_{len(machova_comments)}",
                "content": process_text(row["body"]),
                "troll": 0
            })
    
    # Load and save troll data
    troll_path = data_path / "machova/Is_troll_body.csv"
    if troll_path.exists():
        troll_df = pd.read_csv(troll_path)
        for _, row in troll_df.iterrows():
            machova_comments.append({
                "docId": str(uuid.uuid4()),
                "docType": "Comment",
                "author": f"user_{len(machova_comments)}",
                "content": process_text(row["body"]),
                "troll": 1
            })
    
    with open(output_path / "machova_data.json", "w", encoding="utf-8") as f:
        json.dump({"comments": machova_comments}, f, ensure_ascii=False, indent=4)
    
    logger.info(f"All datasets successfully saved to {output_dir}")

def main():
    """Main function to run the save_data_to_json function with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert raw tweet datasets to JSON format')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the raw data files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory where the JSON files will be saved')
    parser.add_argument('--preprocess', action='store_true',
                      help='Whether to preprocess the text data before saving')
    
    args = parser.parse_args()
    
    try:
        save_data_to_json(args.data_dir, args.output_dir, args.preprocess)
        logger.info("Successfully completed JSON conversion")
    except Exception as e:
        logger.error(f"Error during JSON conversion: {str(e)}")
        raise

if __name__ == "__main__":
    main()
