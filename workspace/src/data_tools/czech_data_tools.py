import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from src.data_tools.preprocessor import TweetPreprocessor

def load_czech_media_data(data_dir: str = "./data/MediaSource") -> pd.DataFrame:
    """Load and process Czech media data from JSON files"""
    data_path = Path(data_dir)
    all_comments = []
    preprocessor = TweetPreprocessor()
    
    # Look for all JSON files
    json_files = list(data_path.glob("*.json"))
    
    for json_file in tqdm(json_files, desc="Loading files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Process each entry in the JSON file
                for entry in data:
                    if entry.get('articleType') == 'Comment':
                        # Preprocess the comment text same as the training data
                        text = preprocessor.preprocess_tweet(entry.get('content', ''))

                        if text.strip():
                            comment_data = {
                                'text': text, # Store preprocessed text
                                'raw_text': entry.get('content', ''), # Keep original for reference
                                'author': entry.get('author', ''),
                                'timestamp': entry.get('publishDate', ''),
                                'article_title': entry.get('title', ''),
                                'url': entry.get('url', ''),
                                'article_id': entry.get('articleId', ''),
                                'sentiment': entry.get('attributes', {}).get('sentiment', ''),
                            }
                            all_comments.append(comment_data)
                        
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(all_comments)
    
    # Convert timestamp to datetime if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    return df

def analyze_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze duplicate comments in the dataset and return cleaned DataFrame
    
    Args:
        df: DataFrame with comment data
        
    Returns:
        DataFrame with duplicate information and statistics
    """
    # Check for exact duplicates
    exact_duplicates = df.duplicated(subset=['text', 'author', 'article_id'], keep=False)
    
    if exact_duplicates.any():
        print("\n=== Duplicate Analysis ===")
        print(f"Total comments: {len(df)}")
        print(f"Duplicate comments: {exact_duplicates.sum()}")
        print(f"Unique comments: {len(df) - exact_duplicates.sum()}")
        
        # Get examples of duplicates
        dup_df = df[exact_duplicates].sort_values(['article_id', 'author', 'timestamp'])
        
        print("\nExample duplicates:")
        print("-" * 80)
        
        # Show a few examples
        for article_id in dup_df['article_id'].unique()[:3]:
            article_dups = dup_df[dup_df['article_id'] == article_id]
            print(f"\nArticle: {article_dups['article_title'].iloc[0]}")
            
            for _, row in article_dups.head(2).iterrows():
                print(f"Author: {row['author']}")
                print(f"Time: {row['timestamp']}")
                print(f"Text: {row['text']}")
                print("-" * 40)
    
        return df.drop_duplicates(subset=['text', 'author', 'article_id'])
    else:
        print("No duplicates found in the dataset")
        return df
    
def show_article_comments(df, n_comments=3, min_comments=2):
    """Show random comments from articles with multiple comments
    
    Args:
        df: DataFrame with comments
        n_comments: Number of random comments to show per article
        min_comments: Minimum number of comments an article should have
    """
    # Drop rows where URL is missing
    df_with_url = df.dropna(subset=['article_url'])
    
    # Get articles with comment counts using URL
    article_comments = df_with_url.groupby('article_url').size()
    
    # Filter for articles with minimum required comments
    eligible_articles = article_comments[article_comments >= min_comments].index
    
    if len(eligible_articles) == 0:
        print("No articles found with enough comments")
        return
    
    # Select random article
    random_article = np.random.choice(eligible_articles)
    
    # Get all comments for this article
    article_df = df_with_url[df_with_url['article_url'] == random_article]
    
    print(f"Article URL: {random_article}")
    print(f"Total comments: {len(article_df)}")
    print("\nRandom sample of comments:")
    
    # Sample random comments
    sample_size = min(n_comments, len(article_df))
    random_comments = article_df.sample(n=sample_size)
    
    for idx, row in random_comments.iterrows():
        print("\n---")
        print(f"Author: {row['author']}")
        print(f"Text: {row['text']}")

def extract_author_comments_simple(df: pd.DataFrame, author: str = None, output_dir: str = "./extracted_comments") -> None:
    """
    Extract comments from a specific author or random author and save only the texts to JSON.
    
    Args:
        df: DataFrame containing the comments
        author: Optional; specific author name to extract. If None, picks random author
        output_dir: Directory to save the JSON file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if author is None:
        # Get authors with at least 2 comments
        author_counts = df['author'].value_counts()
        eligible_authors = author_counts[author_counts >= 2].index
        
        if len(eligible_authors) == 0:
            print("No authors found with multiple comments")
            return
            
        author = np.random.choice(eligible_authors)
        print(f"Randomly selected author: {author}")
    
    # Get author's comments
    author_comments = df[df['author'] == author]
    
    if len(author_comments) == 0:
        print(f"No comments found for author: {author}")
        return
    
    # Extract only the comment texts
    comments = author_comments['text'].tolist()
    
    # Create filename with timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{author.replace(' ', '_')}_{timestamp}.json"
    filepath = Path(output_dir) / filename
    
    # Save to JSON with proper encoding for Czech characters
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)
    
    print(f"\nExtracted {len(comments)} comments from {author}")
    print(f"Comments saved to: {filepath}")
    
    # Show sample of comments
    print("\nSample of comments:")
    print("-" * 80)
    for comment in comments[:6]:
        print(comment)
        print("-" * 80)

def export_comments_by_prediction(predictions_df: pd.DataFrame, 
                              comments_df: pd.DataFrame,
                              prediction_class: str = 'troll',
                              min_confidence: float = 0.9,
                              max_confidence: float = 1.0,
                              max_authors: int = 50,
                              output_file: str = "predicted_comments.json") -> None:
    """
    Export comments sorted by prediction confidence to JSON.

    Args:
        predictions_df: DataFrame with prediction results containing 'author' and 'confidence'
        comments_df: DataFrame with original comments containing 'text', 'author', 'url'
        prediction_class: Class to export ('troll' or 'not_troll')
        min_confidence: Minimum confidence threshold (default: 0.9)
        max_confidence: Maximum confidence threshold (default: 1.0)
        max_authors: Maximum number of authors to include (default: None, includes all)
        output_file: Output JSON file path
    """
    # Validate prediction_class
    if prediction_class not in ['troll', 'not_troll']:
        raise ValueError("prediction_class must be either 'troll' or 'not_troll'")
    
    # Filter high confidence predictions for the specified class
    filtered_predictions = predictions_df[
        (predictions_df['prediction'] == prediction_class) & 
        (predictions_df['confidence'] >= min_confidence) &
        (predictions_df['confidence'] <= max_confidence)
    ].sort_values('confidence', ascending=False)
    
    # Limit number of authors if specified
    if max_authors is not None:
        filtered_predictions = filtered_predictions.head(max_authors)
        
    # Prepare data structure for JSON
    output_data = []
    
    # Process each filtered prediction
    for _, prediction in tqdm(filtered_predictions.iterrows(), 
                            desc=f"Processing {prediction_class} comments",
                            total=len(filtered_predictions)):
        
        # Get all comments from this author
        author_comments = comments_df[comments_df['author'] == prediction['author']]
        
        # Add each comment to the data structure
        for _, comment in author_comments.iterrows():
            output_data.append({
                'author': comment['author'],
                'confidence': float(prediction['confidence']),  # Convert to float for JSON serialization
                'comment': comment['text'],
                'url': comment['url'] if pd.notna(comment['url']) else None
            })
    
    # Save to JSON file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"\nExported {len(output_data)} comments from {len(filtered_predictions)} authors")
    print(f"Output saved to: {output_path}")