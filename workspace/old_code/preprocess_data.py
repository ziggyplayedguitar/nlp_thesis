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
import matplotlib.pyplot as plt

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
        text = str(text).lower()
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

def load_and_clean_data(data_dir: str) -> pd.DataFrame:
    """Load and combine all datasets including Twitter JSON files."""
    data_path = Path(data_dir)
    preprocessor = TweetPreprocessor()
    
    # Load Russian troll tweets
    logger.info("Loading Russian troll tweets...")
    troll_files = list(data_path.glob("russian_troll_tweets/*.csv"))
    troll_tweets = pd.concat([pd.read_csv(f) for f in troll_files])
    troll_tweets = troll_tweets[['author', 'content', 'language']]
    troll_tweets = troll_tweets[troll_tweets['language'] == 'English']
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
    
    # Load Twitter JSON files from "non_troll_politics" folder
    twitter_data = pd.DataFrame()
    json_folder = data_path / "non_troll_politics"
    if json_folder.exists():
        twitter_data = load_twitter_json(json_folder)
    
    # Load Parquet files from "information_operations" folder
    parquet_folder = data_path / "information_operations"
    parquet_files = list(parquet_folder.glob("*.parquet"))
    parquet_data = pd.concat([pd.read_parquet(f) for f in parquet_files])
    parquet_data = parquet_data[['accountid', 'post_text']]
    parquet_data.rename(columns={'accountid': 'account', 'post_text': 'tweet'}, inplace=True)
    parquet_data['troll'] = 1

    # Combine all datasets
    logger.info("Combining datasets...")
    all_tweets = pd.concat([
        troll_tweets[['account', 'tweet', 'troll']],
        sentiment_tweets[['account', 'tweet', 'troll']],
        celeb_tweets[['account', 'tweet', 'troll']],
        twitter_data[['account', 'tweet', 'troll']]
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

def load_and_clean_machova_data(data_dir: str) -> pd.DataFrame:
    """Load and preprocess Czech troll/non-troll comment data."""
    data_path = Path(data_dir)
    preprocessor = TweetPreprocessor()
    
    # Load troll and non-troll data
    logger.info("Loading Czech troll/non-troll data...")
    
    # Load non-troll comments
    nontroll_path = data_path / "machova/Is_not_troll_body.csv"
    nontroll_df = pd.read_csv(nontroll_path)
    logger.info(f"Non-troll data columns: {nontroll_df.columns.tolist()}")
    nontroll_df['troll'] = 0
    
    # Load troll comments
    troll_path = data_path / "machova/Is_troll_body.csv"
    troll_df = pd.read_csv(troll_path)
    logger.info(f"Troll data columns: {troll_df.columns.tolist()}")
    troll_df['troll'] = 1
    
    # Combine datasets
    all_comments = pd.concat([troll_df, nontroll_df], ignore_index=True)
    
    # Create sequential account IDs since we don't have actual account information
    all_comments['account'] = [f"user_{i}" for i in range(len(all_comments))]
    
    # Rename columns to match existing pipeline
    all_comments = all_comments.rename(columns={
        'body': 'tweet'
    })
    
    # Apply preprocessing to comment text
    all_comments['tweet'] = all_comments['tweet'].apply(preprocessor.preprocess_tweet)
    
    # Remove empty comments after preprocessing
    all_comments = all_comments[all_comments['tweet'].str.len() > 0]
    
    logger.info(f"Loaded {len(all_comments)} Czech comments")
    logger.info(f"Class distribution:")
    logger.info(all_comments['troll'].value_counts())
    
    return all_comments

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

def create_data_splits(df: pd.DataFrame, 
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test sets at account level"""
    accounts = df['account'].unique()
    np.random.shuffle(accounts)
    
    n_accounts = len(accounts)
    n_train = int(n_accounts * train_ratio)
    n_val = int(n_accounts * val_ratio)
    
    train_accounts = accounts[:n_train]
    val_accounts = accounts[n_train:n_train + n_val]
    test_accounts = accounts[n_train + n_val:]
    
    train_df = df[df['account'].isin(train_accounts)]
    val_df = df[df['account'].isin(val_accounts)]
    test_df = df[df['account'].isin(test_accounts)]
    
    return train_df, val_df, test_df

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

def load_and_clean_data_json_test(data_dir: str) -> pd.DataFrame:
    """Load and combine all datasets including Twitter JSON files."""
    data_path = Path(data_dir)
    
    # Load Twitter JSON files from "non_troll_politics" folder
    json_folder = data_path / "non_troll_politics"
    twitter_data = pd.DataFrame()
    if json_folder.exists():
        twitter_data = load_twitter_json(json_folder)
    
    return twitter_data

def plot_tweet_lengths(tweets_df: pd.DataFrame):
    """
    Plots the distribution of tweet lengths in the dataset.

    Args:
        tweets_df (pd.DataFrame): DataFrame containing tweets in a column named 'tweet'.
    """
    import matplotlib.pyplot as plt

    # Calculate tweet lengths, handling NaN values gracefully
    tweet_lengths = tweets_df['tweet'].fillna("").astype(str).apply(lambda x: len(x.split()))

    plt.figure(figsize=(10, 6))
    plt.hist(tweet_lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Tweet Lengths')
    plt.xlabel('Tweet Length (number of words)')
    plt.ylabel('Number of Tweets')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import pandas as pd

    data_dir = "./data"

    # Load and preprocess data
    all_tweets = load_and_clean_data(data_dir)
    logger.info(f"Loaded {len(all_tweets)} tweets from {len(all_tweets['account'].unique())} accounts")

    # Plot tweet lengths
    plot_tweet_lengths(all_tweets)

    # The rest of your existing main function
    pass  # (Existing implementation here)


