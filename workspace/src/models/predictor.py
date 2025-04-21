import torch
from transformers import AutoTokenizer
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Union
import pandas as pd

from src.models.bert_model import TrollDetector
from src.data_tools.preprocessor import TweetPreprocessor

class TrollPredictor:
    def __init__(
        self,
        model_path: str = None,  # Optional checkpoint path
        model_name: str = None,  # Optional Hugging Face model name
        device: str = None,
        comments_per_user: int = 5,
        max_length: int = 96
    ):
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize model
        if model_name:
            # Load pretrained model from Hugging Face
            self.model = TrollDetector(model_name=model_name)
        elif model_path:
            # Load model from checkpoint
            self.model = TrollDetector()
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
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
        
    def predict_batch(self, texts: List[str]) -> Dict[str, Union[str, float]]:
        """Simple batch prediction without explanations"""
        inputs = self.prepare_input(texts)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                tweets_per_account=self.comments_per_user
            )
            
            probs = torch.softmax(outputs['logits'], dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()
            
            return {
                'prediction': 'troll' if prediction == 1 else 'not_troll',
                'confidence': confidence,
                'probabilities': probs[0].cpu().numpy()
            }
        
    def preprocess_tweets(self, tweets: List[str]) -> List[str]:
        """Preprocess a list of tweets"""
        return [self.preprocessor.preprocess_tweet(tweet) for tweet in tweets]
    
    def prepare_input(self, tweets: List[str]) -> Dict[str, torch.Tensor]:
        """Prepare input tweets for the model"""
        # Preprocess tweets
        processed_tweets = self.preprocess_tweets(tweets)
        
        # Handle case where we have fewer tweets than required
        if len(processed_tweets) < self.comments_per_user:
            # Repeat tweets to reach desired count
            processed_tweets = (processed_tweets * ((self.comments_per_user // len(processed_tweets)) + 1))[:self.comments_per_user]
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
    
    def predict(self, tweets: List[str]) -> Dict[str, Union[str, float]]:
        """
        Predict whether an account is a troll based on their tweets.
        Now uses batching when there are more tweets than comments_per_user.
        
        Args:
            tweets: List of tweets from the account
                
        Returns:
            Dictionary containing:
                prediction: "troll" or "not_troll"
                confidence: Probability of the predicted class
                attention_weights: Attention weights for each tweet (if using attention model)
        """
        # Use batching if we have more tweets than comments_per_user
        if len(tweets) > self.comments_per_user:
            return self.predict_with_batching(tweets)
            
        # Rest of the existing implementation...
        # Prepare input
        inputs = self.prepare_input(tweets)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                tweets_per_account=self.comments_per_user
            )
            
            # Get probabilities
            probs = torch.softmax(outputs['logits'], dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()
            
            result = {
                'prediction': 'troll' if prediction == 1 else 'not_troll',
                'confidence': confidence,
            }
            
            # Add attention weights if available
            if 'tweet_attention_weights' in outputs:
                result['attention_weights'] = outputs['tweet_attention_weights'][0].cpu().tolist()
            
            return result
    
    def explain_prediction(self, tweets: List[str]) -> Dict[str, List[Dict]]:
        """
        Generate explanations for the prediction using occlusion sensitivity analysis.
        
        Args:
            tweets: List of tweets from the account
            
        Returns:
            Dictionary containing explanation data for each tweet
        """
        print("\nGenerating explanations using occlusion sensitivity...")
        
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
            baseline_probs = torch.softmax(outputs['logits'], dim=-1)
            prediction = torch.argmax(baseline_probs, dim=-1).item()
            class_index = prediction  # Use the predicted class
        
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
                    masked_probs = torch.softmax(outputs['logits'], dim=-1)
                
                # Calculate importance as drop in probability for the predicted class
                importance = baseline_probs[0, class_index].item() - masked_probs[0, class_index].item()
                token_importances[j] = importance
            
            # Create visualization (blue for positive importance, red for negative)
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
            plt.title(f'Token Importance for Tweet {i+1}')
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f"importance_tweet_{i+1}.png"
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
            print(f"Occlusion analysis for Tweet {i+1}:")
            print(f"Text: {tweet}")
            print(f"Plot saved to: {plot_filename}")
            print("Top contributing tokens:")
            for contrib in token_contributions[:10]:
                direction = "supporting" if contrib['contribution'] == 'positive' else "opposing"
                print(f"  - '{contrib['token']}': {contrib['importance']:.4f} ({direction} the prediction)")
        
        return {'explanations': explanation_data}
    
    def explain_with_correlation(self, tweets: List[str]) -> Dict[str, List[Dict]]:
        """
        Alternative explanation method: Use correlation between token presence and model confidence.
        
        Args:
            tweets: List of tweets from the account
            
        Returns:
            Dictionary containing explanation data for each tweet
        """
        print("\nGenerating explanations using token correlation analysis...")
        
        # Prepare input
        inputs = self.prepare_input(tweets)
        processed_tweets = inputs.pop('processed_tweets')
        
        # Get baseline prediction
        result = self.predict(tweets)
        prediction_class = 1 if result['prediction'] == 'troll' else 0
        
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
            token_confidences = {}
            
            # Get base confidence with all tokens
            base_inputs = self.prepare_input([tweet])
            with torch.no_grad():
                outputs = self.model(
                    input_ids=base_inputs['input_ids'],
                    attention_mask=base_inputs['attention_mask'],
                    tweets_per_account=1
                )
                base_probs = torch.softmax(outputs['logits'], dim=-1)
                base_confidence = base_probs[0, prediction_class].item()
            
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
                    mod_probs = torch.softmax(outputs['logits'], dim=-1)
                    mod_confidence = mod_probs[0, prediction_class].item()
                
                # Importance is how much the confidence changes
                importance = base_confidence - mod_confidence
                token_importances[j] = importance
                token_confidences[tokens[j]] = mod_confidence
            
            # Create visualization
            plt.figure(figsize=(12, 4))
            
            # Create a blue-white-red colormap
            colors = [(0.8, 0, 0), (1, 1, 1), (0, 0, 0.8)]  # red, white, blue
            cmap = LinearSegmentedColormap.from_list('rwb', colors, N=100)
            
            # Plot only tokens with calculated importance
            plot_tokens = []
            plot_importances = []
            for j in valid_indices:
                if token_importances[j] != 0:
                    plot_tokens.append(tokens[j])
                    plot_importances.append(token_importances[j])
            
            if not plot_tokens:
                print(f"No significant tokens found for tweet {i+1}, skipping visualization...")
                continue
            
            # Normalize importances for visualization
            max_abs_importance = max(abs(np.min(plot_importances)), abs(np.max(plot_importances)))
            if max_abs_importance > 0:
                normalized_importances = np.array(plot_importances) / max_abs_importance
            else:
                normalized_importances = plot_importances
            
            # Create the bar plot
            plt.bar(range(len(plot_tokens)), plot_importances, 
                   color=[cmap(0.5 + 0.5 * imp) for imp in normalized_importances])
            
            plt.xticks(range(len(plot_tokens)), plot_tokens, rotation=45, ha='right')
            plt.xlabel('Tokens')
            plt.ylabel('Importance Score')
            plt.title(f'Token Importance for Tweet {i+1}')
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f"correlation_tweet_{i+1}.png"
            plt.savefig(plot_filename, bbox_inches='tight')
            plt.close()
            
            # Create contribution data
            token_contributions = []
            for token, importance in zip(plot_tokens, plot_importances):
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
            print(f"Token correlation analysis for Tweet {i+1}:")
            print(f"Text: {tweet}")
            print(f"Plot saved to: {plot_filename}")
            print("Top contributing tokens:")
            for contrib in token_contributions[:10]:
                direction = "supporting" if contrib['contribution'] == 'positive' else "opposing"
                print(f"  - '{contrib['token']}': {contrib['importance']:.4f} ({direction} the prediction)")
                if contrib['token'] in token_confidences:
                    print(f"      Without this token: {token_confidences[contrib['token']]:.4f} confidence")
        
        return {'explanations': explanation_data}

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
        pred = self.predict_batch(author_comments)
        
        if pred['prediction'] != 'troll':
            return {'error': f'Author {author} is not classified as a troll (confidence: {pred["confidence"]:.3f})'}
        
        # Generate explanation
        explanation = self.explain_prediction(author_comments)
        
        # Add prediction info
        explanation['prediction'] = {
            'confidence': pred['confidence'],
            'troll_probability': pred['probabilities'][1]
        }
        
        return explanation

    def predict_with_batching(self, tweets: List[str]) -> Dict[str, Union[str, float]]:
        """
        Predict using multiple batches when there are more tweets than comments_per_user.
        Takes batches of comments_per_user and averages the predictions.
        
        Args:
            tweets: List of tweets from the account
            
        Returns:
            Dictionary with the aggregated prediction
        """
        if len(tweets) <= self.comments_per_user:
            # If we have fewer tweets than the batch size, use the regular predict method
            return self.predict(tweets)
        
        # Process all tweets
        processed_tweets = self.preprocess_tweets(tweets)
        
        # Split into batches of size comments_per_user
        batches = [processed_tweets[i:i + self.comments_per_user] 
                   for i in range(0, len(processed_tweets), self.comments_per_user)]
        
        # Collect predictions for each batch
        all_probs = []
        all_predictions = []
        all_confidences = []
        
        for batch in batches:
            # Handle case where the last batch might be smaller than comments_per_user
            if len(batch) < self.comments_per_user:
                # Repeat tweets to reach desired count
                batch = (batch * ((self.comments_per_user // len(batch)) + 1))[:self.comments_per_user]
            
            # Tokenize
            encodings = self.tokenizer(
                batch,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(
                    input_ids=encodings['input_ids'].to(self.device),
                    attention_mask=encodings['attention_mask'].to(self.device),
                    tweets_per_account=self.comments_per_user
                )
                
                # Get probabilities
                probs = torch.softmax(outputs['logits'], dim=-1)
                prediction = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][prediction].item()
                
                all_probs.append(probs[0].cpu().numpy())
                all_predictions.append(prediction)
                all_confidences.append(confidence)
        
        # Aggregate predictions (average probabilities)
        avg_probs = np.mean(all_probs, axis=0)
        final_prediction = np.argmax(avg_probs)
        final_confidence = avg_probs[final_prediction]
        
        result = {
            'prediction': 'troll' if final_prediction == 1 else 'not_troll',
            'confidence': float(final_confidence),
            'probabilities': avg_probs,
            'num_batches': len(batches)
        }
        
        return result