import torch
from transformers import AutoTokenizer
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Union
import pandas as pd
import logging
from tqdm import tqdm

from src.models.bert_model import TrollDetector
from src.data_tools.preprocessor import TweetPreprocessor

logger = logging.getLogger(__name__)

class TrollPredictor:
    def __init__(
        self,
        model_path: str = None,  # Optional checkpoint path
        model_name: str = None,  # Optional Hugging Face model name
        device: str = None,
        comments_per_user: int = 5,
        max_length: int = 96,
        threshold: float = 0.3,  # Add threshold for binary classification
        adapter_path: str = None,
        adapter_name: str = "czech_comments_mlm",
        use_adapter: bool = False  # New parameter to control adapter usage
    ):
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize model
        if model_name:
            # Load pretrained model from Hugging Face
            self.model = TrollDetector(
                model_name=model_name,
                adapter_path=adapter_path if use_adapter else None
            )
        elif model_path:
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
    
    def predict(self, texts: List[str]) -> Dict[str, Union[str, float]]:
        """
        Predict trolliness for texts, automatically handling batching if needed.
        
        Args:
            texts: List of texts to analyze (comments/tweets)
            
        Returns:
            Dictionary containing:
            - prediction: 'troll' or 'not_troll'
            - trolliness_score: float between 0 and 1
            - binary_confidence: confidence in binary prediction
            - attention_weights: attention weights for each text
            - batch_scores: (if batched) scores for each batch
            - num_batches: (if batched) number of batches processed
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
                
                trolliness_score = outputs['trolliness_score'].squeeze().item()
                all_scores.append(trolliness_score)
                
                # Handle attention weights
                attention = outputs['tweet_attention_weights'].squeeze().cpu().tolist()
                if isinstance(attention, list):
                    all_attention_weights.extend([float(w) for w in attention[:len(batch)]])  # Only keep weights for real texts
                else:
                    all_attention_weights.append(float(attention))
        
        # Aggregate results
        avg_score = np.mean(all_scores)
        
        result = {
            'prediction': 'troll' if avg_score >= self.threshold else 'not_troll',
            'trolliness_score': float(avg_score),
            'binary_confidence': abs(avg_score - 0.5) * 2,
            'attention_weights': all_attention_weights[:len(texts)],  # Only return weights for original texts
            'batch_scores': all_scores,
            'num_batches': len(batches)
        }
        
        return result
    
    # Add this method to the TrollPredictor class
    def explain_author_prediction(self, author: str, comments_df: pd.DataFrame) -> Dict:
        """
        Generate explanation for why an author was classified as a troll.
        
        Args:
            author: Name of the author to explain
            comments_df: DataFrame containing comments data
            
        Returns:
            Dictionary containing explanation data
        """
        # Get author's comments
        author_comments = comments_df[comments_df['author'] == author]['text'].tolist()
        
        if not author_comments:
            return {'error': f'No comments found for author: {author}'}
            
        # Get prediction first
        pred = self.predict(author_comments)
        
        if pred['prediction'] != 'troll':
            return {'error': f'Author {author} is not classified as a troll (confidence: {pred["trolliness_score"]:.3f})'}
        
        # Generate explanation
        explanation = self.explain_prediction(author_comments)
        
        # Add prediction info
        explanation['prediction'] = {
            'trolliness_score': pred['trolliness_score'],
            'binary_confidence': pred['binary_confidence']
        }
        
        return explanation

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
        """Load model weights from checkpoint."""
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
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
        except Exception as e:
            logger.error(f"Error loading checkpoint from {checkpoint_path}: {str(e)}")
            raise

    def explain_prediction(self, tweets: List[str], strategy: str = 'occlusion') -> Dict[str, List[Dict]]:
        """
        Generate explanations for model predictions using different strategies.
        
        Args:
            tweets: List of tweets to explain
            strategy: Explanation strategy to use ('occlusion' or 'correlation')
            
        Returns:
            Dictionary containing explanation data for each tweet
        """
        print(f"\nGenerating explanations using {strategy} analysis...")
        
        # Prepare input
        inputs = self.prepare_input(tweets)
        processed_tweets = inputs.pop('processed_tweets')
        
        # Get baseline prediction
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                tweets_per_account=self.comments_per_user
            )
            baseline_score = outputs['trolliness_score'].squeeze().item()
        
        # Create explanation data structure
        explanation_data = []
        
        # Process each tweet
        for i in range(min(self.comments_per_user, len(processed_tweets))):
            print(f"\nProcessing tweet {i+1}/{min(self.comments_per_user, len(processed_tweets))}")
            
            # Get the original tweet text and tokens
            tweet = processed_tweets[i]
            token_ids = inputs['input_ids'][i].cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            
            # Find non-padding token positions
            mask = inputs['attention_mask'][i].cpu().numpy()
            valid_indices = [j for j, m in enumerate(mask) if m > 0 and tokens[j] not in ['[CLS]', '[SEP]']]
            
            if len(valid_indices) == 0:
                print(f"No valid tokens found in tweet {i+1}, skipping...")
                continue
            
            # Initialize importance scores
            token_importances = np.zeros(len(tokens))
            
            if strategy == 'occlusion':
                # Perform occlusion analysis
                for j in valid_indices:
                    # Create a copy of the input with this token masked
                    masked_input_ids = inputs['input_ids'].clone()
                    masked_input_ids[i, j] = self.tokenizer.mask_token_id  # Replace with [MASK]
                    
                    # Get prediction with masked token
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=masked_input_ids,
                            attention_mask=inputs['attention_mask'],
                            tweets_per_account=self.comments_per_user
                        )
                        masked_score = outputs['trolliness_score'].squeeze().item()
                    
                    # Calculate importance as drop in trolliness score
                    importance = baseline_score - masked_score
                    token_importances[j] = importance
                    
            elif strategy == 'correlation':
                # Get base confidence with all tokens
                base_inputs = self.prepare_input([tweet])
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=base_inputs['input_ids'],
                        attention_mask=base_inputs['attention_mask'],
                        tweets_per_account=1
                    )
                    base_score = outputs['trolliness_score'].squeeze().item()
                
                # For each valid token, create a version without it
                for j in valid_indices:
                    # Skip tokens that are too short or just punctuation
                    if len(tokens[j]) <= 1 and not tokens[j].isalnum():
                        continue
                        
                    # Create a version of the tweet without this token
                    modified_tweet = tweet
                    token_text = tokens[j]
                    if token_text.startswith('##'):
                        token_text = token_text[2:]  # Remove ## prefix for wordpiece tokens
                    
                    # Simple approach: remove the token
                    if token_text in modified_tweet:
                        modified_tweet = modified_tweet.replace(token_text, '')
                    
                    # Get prediction without this token
                    mod_inputs = self.prepare_input([modified_tweet])
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=mod_inputs['input_ids'],
                            attention_mask=mod_inputs['attention_mask'],
                            tweets_per_account=1
                        )
                        modified_score = outputs['trolliness_score'].squeeze().item()
                    
                    # Calculate importance as change in score
                    importance = base_score - modified_score
                    token_importances[j] = importance
            
            # Create visualization
            plt.figure(figsize=(12, 4))
            
            # Create a blue-white-red colormap
            colors = [(0.8, 0, 0), (1, 1, 1), (0, 0, 0.8)]  # red, white, blue
            cmap = LinearSegmentedColormap.from_list('rwb', colors, N=100)
            
            # Plot only non-padding tokens
            valid_tokens = [tokens[j] for j in valid_indices]
            valid_importances = [token_importances[j] for j in valid_indices]
            
            # Normalize importances for visualization
            max_abs_importance = max(abs(np.min(valid_importances)), abs(np.max(valid_importances)))
            if max_abs_importance > 0:
                normalized_importances = valid_importances / max_abs_importance
            else:
                normalized_importances = valid_importances
            
            # Create the bar plot
            bars = plt.bar(range(len(valid_tokens)), valid_importances, 
                          color=[cmap(0.5 + 0.5 * imp) for imp in normalized_importances])
            
            plt.xticks(range(len(valid_tokens)), valid_tokens, rotation=45, ha='right')
            plt.xlabel('Tokens')
            plt.ylabel('Importance Score')
            plt.title(f'Token Importance for Tweet {i+1} ({strategy})')
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f"importance_tweet_{i+1}_{strategy}.png"
            plt.savefig(plot_filename, bbox_inches='tight')
            plt.close()
            
            # Create contribution data
            token_contributions = []
            for j, (token, importance) in enumerate(zip(valid_tokens, valid_importances)):
                # Only include significant contributions
                if abs(importance) > 0.01:
                    token_contributions.append({
                        'token': token,
                        'importance': float(importance),
                        'contribution': 'positive' if importance > 0 else 'negative'
                    })
            
            # Sort tokens by absolute importance
            token_contributions.sort(key=lambda x: abs(x['importance']), reverse=True)
            
            # Add to explanation data
            explanation_data.append({
                'tweet_index': i,
                'tweet_text': tweet,
                'plot_filename': plot_filename,
                'token_contributions': token_contributions[:10]  # Top 10 contributors
            })
            
            # Print explanation
            print(f"{strategy.capitalize()} analysis for Tweet {i+1}:")
            print(f"Text: {tweet}")
            print(f"Plot saved to: {plot_filename}")
            print("Top contributing tokens:")
            for contrib in token_contributions[:10]:
                direction = "supporting" if contrib['contribution'] == 'positive' else "opposing"
                print(f"  - '{contrib['token']}': {contrib['importance']:.4f} ({direction} the prediction)")
        
        return {'explanations': explanation_data}