import torch
from transformers import AutoTokenizer
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Union, Optional
import pandas as pd
import logging
from tqdm import tqdm

from src.models.bert_model import TrollDetector
from src.data_tools.preprocessor import TweetPreprocessor

logger = logging.getLogger(__name__)

class TrollPredictor:
    """A class for making predictions using the TrollDetector model.
    
    This class handles the preprocessing of input texts, batching, and making
    predictions using a trained TrollDetector model.
    
    Attributes:
        device: The device to run the model on (CPU/GPU)
        model: The TrollDetector model instance
        tokenizer: The tokenizer for preprocessing text
        preprocessor: The tweet preprocessor
        comments_per_user: Number of comments to process per user
        max_length: Maximum sequence length for tokenization
        threshold: Threshold for binary classification
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,  # Optional checkpoint path
        model_name: Optional[str] = None,  # Optional Hugging Face model name
        device: Optional[str] = None,
        comments_per_user: int = 5,
        max_length: int = 96,
        threshold: float = 0.3,
        adapter_path: Optional[str] = None,
        adapter_name: str = "czech_comments_mlm",
        use_adapter: bool = False
    ) -> None:
        """Initialize the TrollPredictor.
        
        Args:
            model_path: Path to model checkpoint file
            model_name: Name of Hugging Face model to use
            device: Device to run model on ('cuda' or 'cpu')
            comments_per_user: Number of comments to process per user
            max_length: Maximum sequence length for tokenization
            threshold: Threshold for binary classification
            adapter_path: Path to adapter weights
            adapter_name: Name of the adapter
            use_adapter: Whether to use adapter layers
            
        Raises:
            ValueError: If neither model_path nor model_name is provided
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # # Initialize model
        # if model_name:
        #     # Load pretrained model from Hugging Face
        #     self.model = TrollDetector(
        #         model_name=model_name,
        #         adapter_path=adapter_path if use_adapter else None
        #     )
        if model_path:
            # Load model from checkpoint
            self.model = TrollDetector()
            
            # If using adapter, initialize it before loading checkpoint
            if use_adapter and adapter_path:
                self.model.bert.load_adapter(adapter_path, load_as=adapter_name)
                self.model.bert.set_active_adapters(adapter_name)
                self.model.bert.delete_head("mlm")
            
            self.load_checkpoint(model_path)
        else:
            raise ValueError("Either model_path or model_name must be provided.")
        
        # Move model to the correct device
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize tokenizer and preprocessor
        if model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")  # Default tokenizer
        self.preprocessor = TweetPreprocessor()
        
        # Set parameters
        self.comments_per_user = comments_per_user
        self.max_length = max_length
        self.threshold = threshold  # Add threshold as instance variable 
        
    def prepare_input(self, tweets: List[str]) -> Dict[str, torch.Tensor]:
        """Prepare input tweets for the model"""
        # Preprocess tweets directly using the preprocessor
        processed_tweets = [self.preprocessor.preprocess_tweet(tweet) for tweet in tweets]
        
        # Handle case where we have fewer tweets than required
        if len(processed_tweets) < self.comments_per_user:
            # Pad with empty comments instead of repeating
            num_padding = self.comments_per_user - len(processed_tweets)
            padding_tweets = [""] * num_padding  # Empty strings for padding
            processed_tweets.extend(padding_tweets)
        elif len(processed_tweets) > self.comments_per_user:
            # Take only the first comments_per_user tweets
            processed_tweets = processed_tweets[:self.comments_per_user]
        
        # Tokenize
        encodings = self.tokenizer(
            processed_tweets,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].to(self.device),
            'attention_mask': encodings['attention_mask'].to(self.device),
            'processed_tweets': processed_tweets
        }
    
    def predict(self, texts: List[str]) -> Dict[str, Union[str, float, List[float]]]:
        """Predict trolliness for texts, automatically handling batching if needed.
        
        Args:
            texts: List of texts to analyze (comments/tweets)
            
        Returns:
            Dictionary containing:
                prediction: 'troll' or 'not_troll'
                trolliness_score: float between 0 and 1
                binary_confidence: confidence in binary prediction
                attention_weights: attention weights for each text
                batch_scores: (if batched) scores for each batch
                num_batches: (if batched) number of batches processed
                
        Raises:
            ValueError: If input texts list is empty
        """
        # Process all texts directly using the preprocessor
        processed_texts = [self.preprocessor.preprocess_tweet(text) for text in texts]
        
        # Split into batches if needed
        batches = [processed_texts[i:i + self.comments_per_user] 
                  for i in range(0, len(processed_texts), self.comments_per_user)]
        
        # Collect predictions for each batch
        all_scores = []
        all_attention_weights = []
        
        for batch in batches:
            # Handle smaller batch with padding
            if len(batch) < self.comments_per_user:
                num_padding = self.comments_per_user - len(batch)
                padding_texts = [""] * num_padding
                batch.extend(padding_texts)
            
            # Get prediction for this batch
            encodings = self.tokenizer(
                batch,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=encodings['input_ids'].to(self.device),
                    attention_mask=encodings['attention_mask'].to(self.device),
                    tweets_per_account=self.comments_per_user
                )
                
                # Get raw logits from the model
                raw_logits = outputs['trolliness_score'].squeeze() 
                
                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(raw_logits).item() # .item() if batch size is 1 for this specific call
                all_scores.append(probabilities) # Now storing probabilities
                
                # Handle attention weights
                attention = outputs['tweet_attention_weights'].squeeze().cpu().tolist()
                if isinstance(attention, list):
                    all_attention_weights.extend([float(w) for w in attention[:len(batch)]])  # Only keep weights for real texts
                else:
                    all_attention_weights.append(float(attention))
        
        # Aggregate results (now averaging probabilities)
        avg_probability = np.mean(all_scores) 
        
        result = {
            'prediction': 'troll' if avg_probability >= self.threshold else 'not_troll',
            'trolliness_score': float(avg_probability),
            'binary_confidence': abs(avg_probability - 0.5) * 2,
            'attention_weights': all_attention_weights[:len(texts)],
            'batch_scores': all_scores, 
            'num_batches': len(batches)
        }
        
        return result

    def predict_authors(self, authors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict trolliness scores for multiple authors from a DataFrame.
        """
        results = []
        
        for author in tqdm(authors_df['author'].unique()):
            # Get author's comments
            author_comments = authors_df[authors_df['author'] == author]['text'].tolist()
            
            # Get prediction
            pred = self.predict(author_comments)
            
            # Store results
            results.append({
                'author': author,
                'trolliness_score': pred['trolliness_score'],
                'binary_prediction': pred['prediction'],
                'binary_confidence': pred['binary_confidence'],
                'num_comments': len(author_comments),
                'attention_weights': pred.get('attention_weights', None)
            })
        
        return pd.DataFrame(results)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If there's an error loading the checkpoint
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Handle both raw state dict and wrapped checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Filter out adapter weights if adapter is not initialized
            if not hasattr(self.model.bert, 'active_adapters') or not self.model.bert.active_adapters:
                # Remove adapter-related keys from state dict
                state_dict = {k: v for k, v in state_dict.items() 
                             if not any(adapter_key in k for adapter_key in [
                                 'output_adapters',
                                 'adapter_down',
                                 'adapter_up',
                                 'adapters'])}
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            logger.info(f"Loaded model weights from {checkpoint_path}")
        except FileNotFoundError:
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading checkpoint from {checkpoint_path}: {str(e)}")
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
