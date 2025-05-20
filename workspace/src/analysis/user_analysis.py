"""User analysis module for analyzing predictions on individual users."""

import pandas as pd
from typing import Dict, List
from src.models.predictor import TrollPredictor

def analyze_user_predictions(
    username: str, 
    predictions_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    predictor: TrollPredictor
) -> Dict:
    """Analyze predictions and attention weights for all of a specific user's comments.
    
    Args:
        username: The username to analyze
        predictions_df: DataFrame containing predictions for users
        comments_df: DataFrame containing user comments
        predictor: TrollPredictor instance used for making predictions
        
    Returns:
        Dictionary containing analysis results including:
        - Predicted label
        - Trolliness score
        - Prediction confidence
        - Comment-level analysis with attention weights
        
    Raises:
        ValueError: If no comments are found for the specified user
    """
    # Get all user comments
    user_comments = comments_df[comments_df['author'] == username]['text'].tolist()
    if not user_comments:
        raise ValueError(f"No comments found for user {username}")

    # Get predictions and attention weights for all comments at once
    pred_result = predictor.predict(user_comments)
    attention_weights = pred_result.get('attention_weights', [])
    
    # Get overall metrics from prediction results
    trolliness_score = pred_result['trolliness_score']
    # Convert continuous score to binary prediction using threshold
    predicted_label = "troll" if trolliness_score >= predictor.threshold else "non-troll"
    confidence = abs(trolliness_score - predictor.threshold)  # Distance from threshold as confidence
    
    # Prepare the analysis results
    analysis_results = {
        'username': username,
        'predicted_label': predicted_label,
        'trolliness_score': trolliness_score,
        'prediction_confidence': confidence,
        'num_comments': len(user_comments),
        'comment_analysis': []
    }
    
    # Add detailed comment analysis
    for i, (comment, attn) in enumerate(zip(user_comments, attention_weights)):
        analysis_results['comment_analysis'].append({
            'comment_number': i + 1,
            'text': comment,
            'attention_weight': float(attn)
        })
    
    return analysis_results

def format_analysis_results(analysis_results: Dict) -> str:
    """Format analysis results into a readable string format.
    
    Args:
        analysis_results: Dictionary containing analysis results from analyze_user_predictions
        
    Returns:
        Formatted string containing the analysis results
    """
    output = [
        f"\nAnalysis for user: {analysis_results['username']}",
        "-" * 60,
        f"Predicted Label: {analysis_results['predicted_label']}",
        f"Trolliness Score: {analysis_results['trolliness_score']:.3f}",
        f"Prediction Confidence: {analysis_results['prediction_confidence']:.3f}",
        "-" * 60,
        "\nComment Analysis (with attention weights):",
        "-" * 60
    ]
    
    for comment_data in analysis_results['comment_analysis']:
        output.extend([
            f"\nComment {comment_data['comment_number']}:",
            f"Text: {comment_data['text']}",
            f"Attention weight: {comment_data['attention_weight']:.3f}",
            "-" * 40
        ])
    
    return "\n".join(output) 