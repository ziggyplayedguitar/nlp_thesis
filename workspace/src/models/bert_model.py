import torch
import torch.nn as nn
import torch.nn.functional as F
from adapters import AutoAdapterModel
from transformers import AutoConfig
from typing import Dict, Tuple, Optional, List


class TweetAttention(nn.Module):
    """Module for computing attention weights over tweets."""
    def __init__(self, hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, embeddings: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: Tensor of shape [batch_size, num_tweets, hidden_size]
            padding_mask: Tensor of shape [batch_size, num_tweets] indicating padded tweets
            
        Returns:
            Tuple of (weighted_embedding, attention_weights)
        """
        embeddings = self.dropout(embeddings)
        attention_weights = self.attention(embeddings)  # [batch_size, num_tweets, 1]
        if padding_mask is not None:
            attention_weights = attention_weights.masked_fill(
                padding_mask.unsqueeze(-1) == 0,
                float('-inf')
            )
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize over tweets
        
        # Calculate weighted sum
        weighted_embedding = torch.bmm(
            attention_weights.transpose(1, 2),  # [batch_size, 1, num_tweets]
            embeddings  # [batch_size, num_tweets, hidden_size]
        ).squeeze(1)  # [batch_size, hidden_size]
        
        return weighted_embedding, attention_weights.squeeze(-1)

class EnhancedTweetAttention(nn.Module):
    """Module for computing attention weights over tweets using a more complex attention mechanism, transforming the CLS embeddings through a neural network."""
    def __init__(self, hidden_size: int, attention_hidden_size: int = 128, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, attention_hidden_size),
            nn.ReLU(),
            nn.Linear(attention_hidden_size, hidden_size),
            nn.ReLU()
        )
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, embeddings: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: Tensor of shape [batch_size, num_tweets, hidden_size]
            padding_mask: Tensor of shape [batch_size, num_tweets] indicating padded tweets
            
        Returns:
            Tuple of (weighted_embedding, attention_weights)
        """
        embeddings = self.dropout(embeddings)
        transformed_embeddings = self.transform(embeddings)  # [batch_size, num_tweets, hidden_size]
        attention_scores = self.attention(transformed_embeddings)  # [batch_size, num_tweets, 1]
        if padding_mask is not None:
            attention_scores = attention_scores.masked_fill(
                padding_mask.unsqueeze(-1) == 0,
                float('-inf')
            )
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Calculate weighted sum
        weighted_embedding = torch.bmm(
            attention_weights.transpose(1, 2), 
            embeddings
        ).squeeze(1)  # [batch_size, hidden_size]
        
        return weighted_embedding, attention_weights.squeeze(-1)


class TrollDetector(nn.Module):
    """A BERT-based model for detecting troll behavior in social media posts.
    
    This model uses a pre-trained BERT model with optional adapter layers to analyze
    multiple posts from the same account and predict a continuous trolliness score.
    
    Attributes:
        bert: The base BERT model with optional adapter layers
        config: BERT model configuration
        hidden_size: Size of the hidden layers
        regressor_hidden_size: Size of the hidden layer in the regressor
        tweet_attention: Module for attention over multiple tweets
        account_dropout: Dropout layer for account-level features
        regressor: Neural network for predicting trolliness score
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        dropout_rate: float = 0.1,
        adapter_path: Optional[str] = None,
        hidden_size: Optional[int] = None,
        regressor_hidden_size: Optional[int] = None,
        use_enhanced_attention: bool = False
    ) -> None:
        """Initialize the TrollDetector model.
        
        Args:
            model_name: Name of the pre-trained BERT model to use
            dropout_rate: Dropout rate for regularization
            adapter_path: Optional path to pre-trained adapter weights
            hidden_size: Optional custom hidden size for the model
            regressor_hidden_size: Optional custom hidden size for the regressor
        """
        super().__init__()
        
        # Load pre-trained BERT model
        self.bert = AutoAdapterModel.from_pretrained(model_name)
        self.config = self.bert.config
        
        # Set hidden sizes
        self.hidden_size = hidden_size or self.config.hidden_size
        self.regressor_hidden_size = regressor_hidden_size or self.hidden_size // 2
        
        if adapter_path is not None:
            # Load and activate trained adapter
            adapter_name = "loaded_adapter"
            self.bert.load_adapter(adapter_path, load_as=adapter_name)
            self.bert.set_active_adapters(adapter_name)
            print(f"Loaded and activated adapter from {adapter_path}")
            self.bert.delete_head("mlm")
        
        # Choose attention mechanism
        if use_enhanced_attention:
            self.tweet_attention = EnhancedTweetAttention(self.hidden_size, dropout_rate=dropout_rate)
        else:
            self.tweet_attention = TweetAttention(self.hidden_size, dropout_rate=dropout_rate)
        
        # Layers for account-level regression
        self.account_dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, self.regressor_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.regressor_hidden_size, 1),
        )
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tweets_per_account: int,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            input_ids: Tensor of shape [batch_size * tweets_per_account, seq_length]
            attention_mask: Tensor of shape [batch_size * tweets_per_account, seq_length]
            tweets_per_account: Number of tweets per account in the batch
            padding_mask: Tensor of shape [batch_size * tweets_per_account, seq_length] indicating padded tweets
            
        Returns:
            Dictionary containing:
                trolliness_score: Regression output [batch_size, 1]
                tweet_attention_weights: Attention weights for tweets [batch_size, tweets_per_account]
                account_embedding: Account-level embedding [batch_size, hidden_size]
        """
        batch_size = input_ids.size(0) // tweets_per_account
        
        # Get BERT embeddings
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        
        # Get the last layer hidden states and CLS token embeddings
        cls_embeddings = bert_outputs.hidden_states[-1][:, 0, :] 
        cls_embeddings = cls_embeddings.view(batch_size, tweets_per_account, -1)
        
        # Apply tweet attention with padding mask
        account_embedding, attention_weights = self.tweet_attention(
            cls_embeddings,
            padding_mask=padding_mask
        )
        
        # Apply dropout and regression layers
        account_embedding = self.account_dropout(account_embedding)
        trolliness_score = self.regressor(account_embedding)
        
        return {
            'trolliness_score': trolliness_score,
            'tweet_attention_weights': attention_weights,
            'account_embedding': account_embedding
        }
    
    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tweets_per_account: int,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Make predictions and return trolliness scores.
        
        Args:
            input_ids: Tensor of shape [batch_size * tweets_per_account, seq_length]
            attention_mask: Tensor of shape [batch_size * tweets_per_account, seq_length]
            tweets_per_account: Number of tweets per account
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary containing:
                scores: Continuous scores between 0 and 1 [batch_size]
                binary_predictions: Binary predictions using threshold [batch_size]
                attention_weights: Attention weights for tweets [batch_size, tweets_per_account]
        """
        self.eval()
        outputs = self.forward(input_ids, attention_mask, tweets_per_account)
        
        return {
            'scores': outputs['trolliness_score'].squeeze(-1),
            'binary_predictions': (outputs['trolliness_score'].squeeze(-1) >= threshold).float(),
            'attention_weights': outputs['tweet_attention_weights']
        }