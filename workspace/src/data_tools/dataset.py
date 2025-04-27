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
    """Dataset class for troll detection with continuous trolliness scores"""
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer_name: str = "distilbert-base-multilingual-cased",
        max_length: int = 128,
        comments_per_user: int = 20,
        max_samples_per_author: int = 100,  # Maximum comment batches per author
        text_column: str = None,  # Allow custom text column name
        label_column: str = 'troll',  # Column containing troll labels/scores
        normalize_labels: bool = True  # Whether to normalize labels to [0,1]
    ):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.comments_per_user = comments_per_user
        self.max_samples_per_author = max_samples_per_author
        self.normalize_labels = normalize_labels
        
        # Determine text column name
        if text_column is not None:
            self.text_column = text_column
        elif 'text' in data.columns:
            self.text_column = 'text'
        elif 'tweet' in data.columns:
            self.text_column = 'tweet'
        else:
            raise ValueError("DataFrame must contain either 'text' or 'tweet' column")
        
        logger.info(f"Using '{self.text_column}' as text column")
        
        # Process labels
        if normalize_labels:
            # Get label statistics
            label_min = data[label_column].min()
            label_max = data[label_column].max()
            
            # Check if labels are already in [0,1]
            if label_min >= 0 and label_max <= 1:
                logger.info("Labels are already normalized between 0 and 1")
                self.label_scaler = None
            else:
                logger.info(f"Normalizing labels from [{label_min}, {label_max}] to [0, 1]")
                # Store scaling parameters for later use
                self.label_scaler = {
                    'min': label_min,
                    'max': label_max
                }
        else:
            self.label_scaler = None
        
        # Group comments by author and create multiple batches for each author
        self.samples = []
        author_groups = data.groupby('author')
        
        for author, comments in author_groups:
            # Get the label for this author
            author_label = comments[label_column].iloc[0]
            
            # Normalize label if needed
            if self.label_scaler is not None:
                author_label = (author_label - self.label_scaler['min']) / (self.label_scaler['max'] - self.label_scaler['min'])
            
            if len(comments) <= comments_per_user:
                # If author has fewer comments than needed, just add one sample
                self.samples.append((author, comments, author_label))
            else:
                # If author has more comments, create multiple batches
                num_batches = min(len(comments) // comments_per_user, max_samples_per_author)
                
                # Shuffle the comments to ensure variety in batches
                shuffled_comments = comments.sample(frac=1, random_state=42)
                
                # Create each batch
                for i in range(num_batches):
                    start_idx = i * comments_per_user
                    end_idx = start_idx + comments_per_user
                    batch_comments = shuffled_comments.iloc[start_idx:end_idx]
                    self.samples.append((f"{author}_{i}", batch_comments, author_label))
        
        logger.info(f"Created {len(self.samples)} samples from {len(author_groups)} authors")
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        author, author_comments, label = self.samples[idx]
        
        # Ensure we have exactly comments_per_user comments
        if len(author_comments) < self.comments_per_user:
            # If we have fewer comments than needed, sample with replacement
            author_comments = author_comments.sample(
                n=self.comments_per_user, 
                replace=True,
                random_state=42
            )
        elif len(author_comments) > self.comments_per_user:
            # This shouldn't happen with our initialization, but just in case
            author_comments = author_comments.sample(
                n=self.comments_per_user,
                random_state=42
            )
        
        # Get comment texts
        comments = author_comments[self.text_column].tolist()
        
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
            'label': torch.tensor(label, dtype=torch.float),  # Changed to float for regression
            'author': author
        }

def create_data_splits(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits ensuring authors don't overlap"""
    
    # Get unique authors
    authors = df['author'].unique()
    
    # Create splits
    n_authors = len(authors)
    train_idx = int(n_authors * train_size)
    val_idx = int(n_authors * (train_size + val_size))
    
    # Shuffle authors
    rng = np.random.RandomState(random_state)
    authors = rng.permutation(authors)
    
    # Split authors
    train_authors = authors[:train_idx]
    val_authors = authors[train_idx:val_idx]
    test_authors = authors[val_idx:]
    
    # Create DataFrames
    train_df = df[df['author'].isin(train_authors)]
    val_df = df[df['author'].isin(val_authors)]
    test_df = df[df['author'].isin(test_authors)]
    
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
    authors = [item['author'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels,
        'author': authors
    }

def aggregate_author_predictions(
    predictions: Dict[str, List], 
    method: str = 'mean'
) -> Dict[str, Dict]:
    """
    Aggregate predictions for the same author from multiple comment batches.
    Now handles continuous trolliness scores.
    
    Args:
        predictions: Dictionary where keys are batch IDs and values contain predictions
                    Must have an 'author' field to identify authors
        method: Aggregation method ('mean', 'max', 'min')
    
    Returns:
        Dictionary with aggregated predictions per author
    """
    # Group by base author (removing batch index if present)
    author_preds = {}
    
    for batch_id, pred_data in predictions.items():
        for i, author in enumerate(pred_data['author']):
            # Extract base author name (remove _0, _1, etc. suffixes)
            base_author = author.split('_')[0] if '_' in author else author
            
            if base_author not in author_preds:
                author_preds[base_author] = {
                    'scores': [],
                    'true_label': pred_data['label'][i]
                }
            
            author_preds[base_author]['scores'].append(pred_data['pred'][i])
    
    # Aggregate predictions for each author
    aggregated = {}
    for author, data in author_preds.items():
        if method == 'mean':
            final_score = np.mean(data['scores'])
        elif method == 'max':
            final_score = np.max(data['scores'])
        elif method == 'min':
            final_score = np.min(data['scores'])
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        aggregated[author] = {
            'final_pred': final_score,
            'true_label': data['true_label'],
            'num_batches': len(data['scores'])
        }
    
    return aggregated