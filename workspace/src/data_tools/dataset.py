import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrollDataset(Dataset):
    """Dataset class for troll detection"""
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer_name: str = "distilbert-base-multilingual-cased",
        max_length: int = 128,
        comments_per_user: int = 20
    ):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.comments_per_user = comments_per_user
        
        # Group comments by account
        self.users = list(data.groupby('account'))
        
    def __len__(self) -> int:
        return len(self.users)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        account, account_comments = self.users[idx]
        
        # Sample comments for this account
        if len(account_comments) > self.comments_per_user:
            account_comments = account_comments.sample(n=self.comments_per_user)
        else:
            # If fewer comments than needed, sample with replacement
            account_comments = account_comments.sample(
                n=self.comments_per_user, 
                replace=True
            )
        
        # Get comment texts
        comments = account_comments['text'].tolist()
        
        # Tokenize all comments
        encodings = self.tokenizer(
            comments,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'label': torch.tensor(account_comments['troll'].iloc[0], dtype=torch.long)
        }

def create_data_splits(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits ensuring accounts don't overlap"""
    
    # Get unique accounts
    accounts = df['account'].unique()
    
    # Create splits
    n_accounts = len(accounts)
    train_idx = int(n_accounts * train_size)
    val_idx = int(n_accounts * (train_size + val_size))
    
    # Shuffle accounts
    rng = np.random.RandomState(random_state)
    accounts = rng.permutation(accounts)
    
    # Split accounts
    train_accounts = accounts[:train_idx]
    val_accounts = accounts[train_idx:val_idx]
    test_accounts = accounts[val_idx:]
    
    # Create DataFrames
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