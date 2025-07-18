import pandas as pd
import numpy as np
import json
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
import matplotlib.pyplot as pl
import uuid
import json
from nltk.corpus import stopwords
import nltk
from datasets import load_dataset
from data.stop_words import STOP_WORDS as CZECH_STOP_WORDS  # Import Czech stop words from your file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TweetPreprocessor:
    """A class for preprocessing social media posts.
    
    This class handles various preprocessing tasks for social media posts including:
    - URL removal
    - Twitter-specific cleanup (mentions, hashtags)
    - Emoji removal
    - Whitespace normalization
    - Stop word removal (optional)
    
    Attributes:
        url_pattern: Regex pattern for matching URLs
        pic_pattern: Regex pattern for matching Twitter picture URLs
        multiple_whitespace: Regex pattern for matching multiple whitespace
        hashtag_pattern: Regex pattern for matching hashtags
        mention_pattern: Regex pattern for matching mentions
        quote_pattern: Regex pattern for matching quoted text
        emoji_pattern: Regex pattern for matching emojis
        stop_words: Set of stop words to remove (optional)
    """
    
    def __init__(self) -> None:
        """Initialize the TweetPreprocessor with regex patterns and stop words."""
        # Initialize regex patterns
        self.url_pattern = re.compile(r'https?://\S+\b')
        self.pic_pattern = re.compile(r'pic\.twitter\.com/\w+\b')
        self.multiple_whitespace = re.compile(r'\s+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.mention_pattern = re.compile(r'@\w+')
        self.quote_pattern = re.compile(r'"([^"]*)"')
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
        # Download NLTK stop words if not already downloaded
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Combine custom Twitter stop words
        self.custom_stop_words = {
            't', 's', 'm', 'rt', 'http', 'https',  # Common Twitter artifacts
            'amp', 'co', 've', 'll', 'd', 're',    # Contractions and common artifacts
            'twitter', 'tweet', 'retweet'         # Twitter-specific terms
        }
        
        # Combine English stop words from NLTK and Czech stop words from file
        self.stop_words = set(stopwords.words('english')).union(
            CZECH_STOP_WORDS
        ).union(self.custom_stop_words)

    def preprocess_tweet(self, text: str) -> str:
        """Preprocess a single tweet.
        
        Args:
            text: The tweet text to preprocess
            
        Returns:
            Preprocessed tweet text with URLs, mentions, hashtags, and emojis removed
        """
        if pd.isna(text):
            return ""
        
        # Remove URLs, Twitter pics, hashtags, mentions, and emojis
        text = self.url_pattern.sub('', text) 
        text = self.pic_pattern.sub('', text)
        text = self.hashtag_pattern.sub('', text)
        text = self.mention_pattern.sub('', text)
        text = self.quote_pattern.sub('', text)
        text = self.emoji_pattern.sub('', text)
        
        # Normalize whitespace
        text = self.multiple_whitespace.sub(' ', text)
        
        return text.strip()

def load_twitter_json(json_folder: Path) -> pd.DataFrame:
    """Load and preprocess tweets from all JSON files in the specified folder.
    
    Args:
        json_folder: Path to folder containing Twitter JSON files
        
    Returns:
        DataFrame containing processed tweets with columns:
            - account: Twitter account name
            - tweet: Tweet text
            - troll: Binary label (0 for non-troll)
    """
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

def limit_tweets_per_author(df: pd.DataFrame, max_tweets_per_author: Optional[int] = None) -> pd.DataFrame:
    """Limit the number of tweets per author in a dataframe.
    
    Args:
        df: DataFrame containing tweets
        max_tweets_per_author: Maximum number of tweets to keep per author
        
    Returns:
        DataFrame with limited tweets per author
    """
    if max_tweets_per_author is None:
        return df
    return df.groupby('account').apply(
        lambda x: x.sample(n=min(len(x), max_tweets_per_author), random_state=42)
    ).reset_index(drop=True)

def load_and_clean_data(
    data_dir: str,
    max_tweets_per_source: Optional[int] = None,
    max_tweets_per_author: Optional[int] = None
) -> pd.DataFrame:
    """Load and combine all datasets including Twitter JSON files.
    
    Args:
        data_dir: Directory containing the data
        max_tweets_per_source: Maximum number of tweets to load from each source
        max_tweets_per_author: Maximum number of tweets to keep per author
        
    Returns:
        DataFrame containing combined and cleaned tweets with columns:
            - account: Twitter account name
            - tweet: Tweet text
            - troll: Binary label (1 for troll, 0 for non-troll)
            - language: Language code
    """
    data_path = Path(data_dir)
    preprocessor = TweetPreprocessor()
    
    # Load Russian troll tweets
    logger.info("Loading Russian troll tweets...")
    troll_files = list(data_path.glob("russian_troll_tweets/*.csv"))
    if max_tweets_per_source:
        troll_tweets = []
        for f in troll_files:
            df = pd.read_csv(f)
            if len(df) > max_tweets_per_source:
                df = df.sample(n=max_tweets_per_source, random_state=42)
            troll_tweets.append(df)
        troll_tweets = pd.concat(troll_tweets)
    else:
        troll_tweets = pd.concat([pd.read_csv(f) for f in troll_files])
    troll_tweets = troll_tweets[['author', 'content', 'language']]
    troll_tweets.rename(columns={'author': 'account', 'content': 'tweet'}, inplace=True)
    troll_tweets['troll'] = 1
    troll_tweets = limit_tweets_per_author(troll_tweets, max_tweets_per_author)
    
    # Load Sentiment140 tweets
    logger.info("Loading Sentiment140 tweets...")
    sentiment_path = data_path / "sentiment_tweets/training.1600000.processed.noemoticon.csv"
    if max_tweets_per_source:
        sentiment_tweets = pd.read_csv(sentiment_path, encoding='Latin-1',
                                     names=['target', 'id', 'date', 'flag', 'username', 'tweet'],
                                     nrows=max_tweets_per_source)
    else:
        sentiment_tweets = pd.read_csv(sentiment_path, encoding='Latin-1',
                                     names=['target', 'id', 'date', 'flag', 'username', 'tweet'])
    sentiment_tweets = sentiment_tweets[['username', 'tweet']]
    sentiment_tweets.rename(columns={'username': 'account'}, inplace=True)
    sentiment_tweets['troll'] = 0
    sentiment_tweets['language'] = 'en'
    sentiment_tweets = limit_tweets_per_author(sentiment_tweets, max_tweets_per_author)

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
    celeb_tweets = limit_tweets_per_author(celeb_tweets, max_tweets_per_author)
    
    # Load Twitter JSON files from "non_troll_politics" folder
    logger.info("Loading manualy scraped tweets...")
    twitter_data = pd.DataFrame()
    json_folder = data_path / "non_troll_politics"
    if json_folder.exists():
        twitter_data = load_twitter_json(json_folder)
        twitter_data['language'] = 'en'
        twitter_data = limit_tweets_per_author(twitter_data, max_tweets_per_author)
    
    # Load Parquet files from "information_operations" folder
    logger.info("Loading information operations tweets...")
    parquet_folders = [data_path / "information_operations/Russia", data_path / "information_operations/Spain"]
    parquet_files = []
    for folder in parquet_folders:
        if folder.exists():
            parquet_files.extend(list(folder.glob("*.parquet")))
    logger.info(f"Found {len(parquet_files)} parquet files in information_operations folder and its subdirectories")
    
    if max_tweets_per_source:
        parquet_dfs = []
        for f in parquet_files:
            df = pd.read_parquet(f)
            if len(df) > max_tweets_per_source:
                df = df.sample(n=max_tweets_per_source, random_state=42)
            parquet_dfs.append(df)
        parquet_data = pd.concat(parquet_dfs)
    else:
        parquet_data = pd.concat([pd.read_parquet(f) for f in parquet_files])
    parquet_data = parquet_data[['accountid', 'post_text', 'is_control', 'post_language']]
    parquet_data.rename(columns={
        'accountid': 'account',
        'post_text': 'tweet',
        'post_language': 'language'
    }, inplace=True)
    parquet_data['troll'] = ~parquet_data['is_control']  # troll is True when is_control is False
    logger.info(f"Information operations data distribution - Trolls: {parquet_data['troll'].sum()}, Non-trolls: {(~parquet_data['troll']).sum()}")
    parquet_data = parquet_data.drop('is_control', axis=1)
    parquet_data = limit_tweets_per_author(parquet_data, max_tweets_per_author)

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
    combined_df['language'] = 'en'
    combined_df = limit_tweets_per_author(combined_df, max_tweets_per_author)

    # # Load Civil Comments dataset
    # logger.info("Loading Civil Comments dataset...")
    # civil_comments = load_dataset("google/civil_comments")
    
    # # Load all splits and combine them
    # civil_train = pd.DataFrame(civil_comments['train'])
    # civil_test = pd.DataFrame(civil_comments['test'])
    # civil_val = pd.DataFrame(civil_comments['validation'])
    
    # # Combine all splits
    # civil_df = pd.concat([civil_train, civil_test, civil_val], ignore_index=True)
    # # Filter for low toxicity comments
    # civil_df = civil_df[civil_df['toxicity'] <= 0.1]
    # # Create artificial authors by grouping comments
    # # Group size of 10 comments per author
    # GROUP_SIZE = 10
    # civil_df = civil_df.sort_index()  # Sort by index to ensure consistent grouping
    # civil_df['group_id'] = civil_df.index // GROUP_SIZE
    # civil_df['account'] = 'civil_author_' + civil_df['group_id'].astype(str)
    
    # # Select and rename columns
    # civil_df = civil_df[['text', 'account']]
    # civil_df.rename(columns={
    #     'text': 'tweet'
    # }, inplace=True)
    # civil_df['troll'] = 0
    # civil_df['language'] = 'en'

    # Combine all datasets
    logger.info("Combining datasets...")
    all_tweets = pd.concat([
        troll_tweets[['account', 'tweet', 'troll', 'language']],
        sentiment_tweets[['account', 'tweet', 'troll', 'language']],
        celeb_tweets[['account', 'tweet', 'troll', 'language']],
        twitter_data[['account', 'tweet', 'troll', 'language']],
        parquet_data[['account', 'troll', 'tweet', 'language']],
        combined_df[['account', 'tweet', 'troll', 'language']],
        # civil_df[['account', 'tweet', 'troll', 'language']]
    ], ignore_index=True)

    # Apply preprocessing to tweet text
    all_tweets['tweet'] = all_tweets['tweet'].apply(preprocessor.preprocess_tweet)

    # Remove empty tweets after preprocessing
    all_tweets = all_tweets[all_tweets['tweet'].str.len() > 0]
    
    # Remove accounts with very few tweets
    logger.info("Filtering accounts with few tweets...")
    account_counts = all_tweets.groupby('account').size()
    valid_accounts = account_counts[account_counts >= 5].index
    all_tweets = all_tweets[all_tweets['account'].isin(valid_accounts)]
    
    return all_tweets

class TrollTweetDataset(Dataset):
    """Dataset class for troll detection using tweets.
    
    This dataset handles loading and preprocessing of tweets for training a
    troll detection model. It supports batching of multiple tweets per account.
    
    Attributes:
        tweets_df: DataFrame containing the raw tweets
        tokenizer: Tokenizer for text preprocessing
        max_length: Maximum sequence length for tokenization
        tweets_per_account: Number of tweets to process per account
        samples: List of processed samples
    """
    
    def __init__(
        self,
        tweets_df: pd.DataFrame,
        tokenizer_name: str = "distilbert-base-multilingual-cased",
        max_length: int = 128,
        tweets_per_account: int = 5
    ) -> None:
        """Initialize the TrollTweetDataset.
        
        Args:
            tweets_df: DataFrame containing tweets with columns:
                - account: Twitter account name
                - tweet: Tweet text
                - troll: Binary label
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length for tokenization
            tweets_per_account: Number of tweets to process per account
        """
        self.tweets_df = tweets_df
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.tweets_per_account = tweets_per_account
        
        # Create samples
        self.samples = []
        for account, group in tweets_df.groupby('account'):
            tweets = group['tweet'].tolist()
            label = group['troll'].iloc[0]
            
            # Handle case where we have fewer tweets than required
            if len(tweets) < tweets_per_account:
                tweets.extend([''] * (tweets_per_account - len(tweets)))
            elif len(tweets) > tweets_per_account:
                tweets = tweets[:tweets_per_account]
            
            self.samples.append((tweets, label, account))
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Dictionary containing:
                input_ids: Tokenized tweet IDs
                attention_mask: Attention mask for the tokens
                label: Binary label for troll detection
                account: Account identifier
        """
        tweets, label, account = self.samples[idx]
        
        # Tokenize tweets
        encodings = self.tokenizer(
            tweets,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float),
            'account': account
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
    accounts = [item['account'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels,
        'account': accounts
    }

