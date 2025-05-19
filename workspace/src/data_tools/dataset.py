import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from tqdm import tqdm
import hashlib
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrollDataset(Dataset):
    """Dataset class for troll detection with continuous trolliness scores"""
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer_name: str = "distilbert-base-multilingual-cased",
        max_length: int = 128,
        comments_per_user: int = 10,
        max_samples_per_author: int = 100,
        label_column: str = 'troll',
        normalize_labels: bool = True,
        cache_dir: Optional[str] = None,
        use_dynamic_padding: bool = False
    ):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.comments_per_user = comments_per_user
        self.max_samples_per_author = max_samples_per_author
        self.normalize_labels = normalize_labels
        self.use_dynamic_padding = use_dynamic_padding
        
        # Set up caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = self.cache_dir / "tokenized_data.json"
            self.cache = self._load_cache()
        else:
            self.cache = {}
        
        # Validate input data
        self._validate_data()
        
        # Process labels
        self._process_labels(label_column)
        
        # Create samples with progress bar
        self.samples = []
        author_groups = data.groupby('author')
        
        for author, comments in tqdm(author_groups, desc="Creating samples"):
            self._create_author_samples(author, comments, label_column)
        
        logger.info(f"Created {len(self.samples)} samples from {len(author_groups)} authors")
    
    def _validate_data(self) -> None:
        """Validate input data for required columns and data types."""
        required_columns = ['author', 'text']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for empty texts
        empty_texts = self.data['text'].isna().sum()
        if empty_texts > 0:
            logger.warning(f"Found {empty_texts} empty texts in the dataset")
            self.data = self.data.dropna(subset=['text'])
    
    def _process_labels(self, label_column: str) -> None:
        """Process and normalize labels."""
        if self.normalize_labels:
            label_min = self.data[label_column].min()
            label_max = self.data[label_column].max()
            
            if label_min >= 0 and label_max <= 1:
                logger.info("Labels are already normalized between 0 and 1")
                self.label_scaler = None
            else:
                logger.info(f"Normalizing labels from [{label_min}, {label_max}] to [0, 1]")
                self.label_scaler = {
                    'min': label_min,
                    'max': label_max
                }
        else:
            self.label_scaler = None
    
    def _create_author_samples(self, author: str, comments: pd.DataFrame, label_column: str) -> None:
        """Create samples for a single author."""
        author_label = comments[label_column].iloc[0]
        
        if self.label_scaler is not None:
            author_label = (author_label - self.label_scaler['min']) / (self.label_scaler['max'] - self.label_scaler['min'])
        
        if len(comments) <= self.comments_per_user:
            self.samples.append((author, comments, author_label))
        else:
            num_batches = min(len(comments) // self.comments_per_user, self.max_samples_per_author)
            shuffled_comments = comments.sample(frac=1, random_state=42)
            
            for i in range(num_batches):
                start_idx = i * self.comments_per_user
                end_idx = start_idx + self.comments_per_user
                batch_comments = shuffled_comments.iloc[start_idx:end_idx]
                self.samples.append((f"{author}_{i}", batch_comments, author_label))
    
    def _get_cache_key(self, comments: List[str]) -> str:
        """Generate a cache key for a list of comments."""
        return hashlib.md5(json.dumps(comments, sort_keys=True).encode()).hexdigest()
    
    def _load_cache(self) -> Dict[str, Dict]:
        """Load cached tokenized data."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self) -> None:
        """Save tokenized data to cache."""
        if self.cache_dir:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache, f)
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        author, author_comments, label = self.samples[idx]
        
        # Get comment texts
        comments = author_comments['text'].tolist()
        
        # Handle padding
        if len(comments) < self.comments_per_user:
            comments.extend([""] * (self.comments_per_user - len(comments)))
        elif len(comments) > self.comments_per_user:
            comments = comments[:self.comments_per_user]
        
        # Check cache
        cache_key = self._get_cache_key(comments)
        if cache_key in self.cache:
            encodings = self.cache[cache_key]
            input_ids = torch.tensor(encodings['input_ids'])
            attention_mask = torch.tensor(encodings['attention_mask'])
        else:
            # Tokenize with appropriate padding
            padding = 'max_length' if not self.use_dynamic_padding else 'longest'
            encodings = self.tokenizer(
                comments,
                padding=padding,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Cache the results
            if self.cache_dir:
                self.cache[cache_key] = {
                    'input_ids': encodings['input_ids'].tolist(),
                    'attention_mask': encodings['attention_mask'].tolist()
                }
                self._save_cache()
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'label': torch.tensor(label, dtype=torch.float),
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


def aggregate_author_predictions(predictions: Dict[str, List], method: str = 'mean') -> Dict[str, Dict]:
    author_preds = {}
    
    for batch_id, pred_data in predictions.items():
        # Skip empty batches
        if len(pred_data['author']) == 0:
            continue
            
        for i, author in enumerate(pred_data['author']):
            # Validate indices
            if i >= len(pred_data['label']) or i >= len(pred_data['pred']):
                continue
                
            base_author = author.split('_')[0] if '_' in author else author
            
            if base_author not in author_preds:
                author_preds[base_author] = {
                    'scores': [],
                    'true_label': pred_data['label'][i]
                }
            
            author_preds[base_author]['scores'].append(pred_data['pred'][i])
    
    # Only aggregate if we have predictions
    aggregated = {}
    for author, data in author_preds.items():
        if not data['scores']:  # Skip if no scores
            continue
            
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
