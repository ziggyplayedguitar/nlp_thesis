import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig
from typing import Dict, Tuple

class TrollDetector(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        dropout_rate: float = 0.1,
        freeze_bert: bool = False
    ):
        super().__init__()
        
        # Load pre-trained DistilBERT
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.config = self.bert.config
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Layers for tweet-level processing
        self.tweet_dropout = nn.Dropout(dropout_rate)
        self.tweet_attention = nn.Linear(self.config.hidden_size, 1)
        
        # Layers for account-level classification
        self.account_dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.config.hidden_size // 2, 2)  # Binary classification
        )
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tweets_per_account: int
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Tensor of shape [batch_size * tweets_per_account, seq_length]
            attention_mask: Tensor of shape [batch_size * tweets_per_account, seq_length]
            tweets_per_account: Number of tweets per account in the batch
            
        Returns:
            Dictionary containing:
                logits: Classification logits [batch_size, 2]
                tweet_attention_weights: Attention weights for tweets [batch_size, tweets_per_account]
        """
        batch_size = input_ids.size(0) // tweets_per_account
        
        # Get BERT embeddings
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get CLS token embeddings [batch_size * tweets_per_account, hidden_size]
        cls_embeddings = bert_outputs.last_hidden_state[:, 0, :]
        cls_embeddings = self.tweet_dropout(cls_embeddings)
        
        # Reshape to [batch_size, tweets_per_account, hidden_size]
        cls_embeddings = cls_embeddings.view(batch_size, tweets_per_account, -1)
        
        # Calculate attention weights for tweets
        attention_weights = self.tweet_attention(cls_embeddings)  # [batch_size, tweets_per_account, 1]
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize over tweets
        
        # Calculate weighted sum of tweet embeddings
        account_embedding = torch.bmm(
            attention_weights.transpose(1, 2),  # [batch_size, 1, tweets_per_account]
            cls_embeddings  # [batch_size, tweets_per_account, hidden_size]
        ).squeeze(1)  # [batch_size, hidden_size]
        
        # Apply dropout and classification layers
        account_embedding = self.account_dropout(account_embedding)
        logits = self.classifier(account_embedding)
        
        return {
            'logits': logits,
            'tweet_attention_weights': attention_weights.squeeze(-1)
        }
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tweets_per_account: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions and return class probabilities.
        
        Args:
            Same as forward()
            
        Returns:
            predictions: Class predictions [batch_size]
            probabilities: Class probabilities [batch_size, 2]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, tweets_per_account)
            probabilities = F.softmax(outputs['logits'], dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        return predictions, probabilities

class TrollDetectorWithPooling(nn.Module):
    """Alternative version using simple pooling instead of attention"""
    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        dropout_rate: float = 0.1,
        freeze_bert: bool = False,
        pooling_type: str = "mean"
    ):
        super().__init__()
        
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.config = self.bert.config
        self.pooling_type = pooling_type
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.config.hidden_size // 2, 2)
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tweets_per_account: int
    ) -> Dict[str, torch.Tensor]:
        batch_size = input_ids.size(0) // tweets_per_account
        
        # Get BERT embeddings
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get CLS token embeddings and reshape
        cls_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # [batch_size * tweets_per_account, hidden_size]
        cls_embeddings = cls_embeddings.view(batch_size, tweets_per_account, -1)  # [batch_size, tweets_per_account, hidden_size]
        
        # Apply pooling
        if self.pooling_type == "mean":
            account_embedding = cls_embeddings.mean(dim=1)  # [batch_size, hidden_size]
        elif self.pooling_type == "max":
            account_embedding = cls_embeddings.max(dim=1)[0]  # [batch_size, hidden_size]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        # Classification
        account_embedding = self.dropout(account_embedding)
        logits = self.classifier(account_embedding)
        
        return {
            'logits': logits,
            'pooled_embedding': account_embedding
        }
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tweets_per_account: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, tweets_per_account)
            probabilities = F.softmax(outputs['logits'], dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        return predictions, probabilities

# Example usage and test code
if __name__ == "__main__":
    # Create dummy data
    batch_size = 2
    tweets_per_account = 5
    seq_length = 128
    
    input_ids = torch.randint(
        0, 1000, 
        (batch_size * tweets_per_account, seq_length)
    )
    attention_mask = torch.ones_like(input_ids)
    
    # Test attention-based model
    print("Testing TrollDetector (attention-based)...")
    model = TrollDetector()
    outputs = model(input_ids, attention_mask, tweets_per_account)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Attention weights shape: {outputs['tweet_attention_weights'].shape}")
    
    # Test prediction
    predictions, probs = model.predict(input_ids, attention_mask, tweets_per_account)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probs.shape}")
    
    # Test pooling-based model
    print("\nTesting TrollDetector (pooling-based)...")
    model_pooling = TrollDetectorWithPooling(pooling_type="mean")
    outputs = model_pooling(input_ids, attention_mask, tweets_per_account)
    print(f"Logits shape: {outputs['logits'].shape}")
    
    predictions, probs = model_pooling.predict(input_ids, attention_mask, tweets_per_account)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probs.shape}")