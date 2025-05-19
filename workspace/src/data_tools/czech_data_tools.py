import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging
from typing import List, Dict, Optional
from src.data_tools.preprocessor import TweetPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_czech_media_data(data_dir: str = "./data/MediaSource") -> pd.DataFrame:
    """Load and process Czech media data from JSON files."""
    data_path = Path(data_dir)
    all_comments = []
    preprocessor = TweetPreprocessor()
    
    # Look for all JSON files
    json_files = list(data_path.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    for json_file in tqdm(json_files, desc="Loading files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for entry in data:
                    if entry.get('articleType') == 'Comment':
                        try:
                            text = preprocessor.preprocess_tweet(entry.get('content', ''))
                            
                            if text.strip():
                                comment_data = {
                                    'text': text,
                                    'raw_text': entry.get('content', ''),
                                    'author': entry.get('author', ''),
                                    'timestamp': entry.get('publishDate', ''),
                                    'article_title': entry.get('title', ''),
                                    'url': entry.get('url', ''),
                                    'article_id': entry.get('articleId', ''),
                                    'sentiment': entry.get('attributes', {}).get('sentiment', ''),
                                }
                                all_comments.append(comment_data)
                        except Exception as e:
                            logger.warning(f"Error processing comment in {json_file}: {e}")
                            continue
                            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {json_file}: {e}")
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error in {json_file}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing {json_file}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(all_comments)
    
    # Convert timestamp to datetime if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    return df

def analyze_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze duplicate comments in the dataset and return cleaned DataFrame."""
    # Check for exact duplicates
    exact_duplicates = df.duplicated(subset=['text', 'author', 'article_id'], keep=False)
    
    if exact_duplicates.any():
        logger.info("\n=== Duplicate Analysis ===")
        logger.info(f"Total comments: {len(df)}")
        logger.info(f"Duplicate comments: {exact_duplicates.sum()}")
        logger.info(f"Unique comments: {len(df) - exact_duplicates.sum()}")
        
        # Get examples of duplicates
        dup_df = df[exact_duplicates].sort_values(['article_id', 'author', 'timestamp'])
        
        logger.info("\nExample duplicates:")
        logger.info("-" * 80)
        
        # Show a few examples
        for article_id in dup_df['article_id'].unique()[:3]:
            article_dups = dup_df[dup_df['article_id'] == article_id]
            logger.info(f"\nArticle: {article_dups['article_title'].iloc[0]}")
            
            for _, row in article_dups.head(2).iterrows():
                logger.info(f"Author: {row['author']}")
                logger.info(f"Time: {row['timestamp']}")
                logger.info(f"Text: {row['text']}")
                logger.info("-" * 40)
    
        return df.drop_duplicates(subset=['text', 'author', 'article_id'])
    else:
        logger.info("No duplicates found in the dataset")
        return df

def show_article_comments(
    df: pd.DataFrame,
    n_comments: int = 3,
    min_comments: int = 2,
    max_length: Optional[int] = None
) -> None:
    """Show random comments from articles with multiple comments."""
    # Drop rows where URL is missing
    df_with_url = df.dropna(subset=['url'])
    
    # Get articles with comment counts using URL
    article_comments = df_with_url.groupby('url').size()
    
    # Filter for articles with minimum required comments
    eligible_articles = article_comments[article_comments >= min_comments].index
    
    if len(eligible_articles) == 0:
        logger.info("No articles found with enough comments")
        return
    
    # Select random article
    random_article = np.random.choice(eligible_articles)
    
    # Get all comments for this article
    article_df = df_with_url[df_with_url['url'] == random_article]
    
    logger.info(f"Article URL: {random_article}")
    logger.info(f"Total comments: {len(article_df)}")
    logger.info("\nRandom sample of comments:")
    
    # Sample random comments
    sample_size = min(n_comments, len(article_df))
    random_comments = article_df.sample(n=sample_size)
    
    for idx, row in random_comments.iterrows():
        logger.info("\n---")
        logger.info(f"Author: {row['author']}")
        text = row['text']
        if max_length and len(text) > max_length:
            text = text[:max_length] + "..."
        logger.info(f"Text: {text}")

def extract_author_comments_simple(
    df: pd.DataFrame,
    author: Optional[str] = None,
    output_dir: str = "./extracted_comments",
    min_comments: int = 2,
    max_comments: Optional[int] = None
) -> None:
    """Extract comments from a specific author or random author and save to JSON."""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if author is None:
        # Get authors with minimum required comments
        author_counts = df['author'].value_counts()
        eligible_authors = author_counts[author_counts >= min_comments].index
        
        if len(eligible_authors) == 0:
            logger.info("No authors found with multiple comments")
            return
            
        author = np.random.choice(eligible_authors)
        logger.info(f"Randomly selected author: {author}")
    
    # Get author's comments
    author_comments = df[df['author'] == author]
    
    if len(author_comments) == 0:
        logger.info(f"No comments found for author: {author}")
        return
    
    # Limit number of comments if specified
    if max_comments and len(author_comments) > max_comments:
        author_comments = author_comments.sample(n=max_comments, random_state=42)
    
    # Extract only the comment texts
    comments = author_comments['text'].tolist()
    
    # Create filename with timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{author.replace(' ', '_')}_{timestamp}.json"
    filepath = output_path / filename
    
    # Save to JSON with proper encoding for Czech characters
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comments, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\nExtracted {len(comments)} comments from {author}")
        logger.info(f"Comments saved to: {filepath}")
        
        # Show sample of comments
        logger.info("\nSample of comments:")
        logger.info("-" * 80)
        for comment in comments[:6]:
            logger.info(comment)
            logger.info("-" * 80)
    except Exception as e:
        logger.error(f"Error saving comments to {filepath}: {e}")

def export_comments_by_prediction(
    predictions_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    prediction_class: str = 'troll',
    min_confidence: float = 0.9,
    max_confidence: float = 1.0,
    max_authors: int = 50,
    output_file: str = "predicted_comments.json"
) -> None:
    """Export comments sorted by prediction confidence to JSON."""
    try:
        # Filter predictions by class and confidence
        filtered_preds = predictions_df[
            (predictions_df['prediction'] == prediction_class) &
            (predictions_df['confidence'] >= min_confidence) &
            (predictions_df['confidence'] <= max_confidence)
        ]
        
        if len(filtered_preds) == 0:
            logger.warning("No predictions found matching the criteria")
            return
        
        # Sort by confidence and limit authors
        top_preds = filtered_preds.sort_values('confidence', ascending=False).head(max_authors)
        
        # Get corresponding comments
        comments = []
        for _, pred in top_preds.iterrows():
            author_comments = comments_df[comments_df['author'] == pred['author']]
            if not author_comments.empty:
                comments.extend(author_comments['text'].tolist())
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comments, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported {len(comments)} comments to {output_file}")
        
    except Exception as e:
        logger.error(f"Error exporting comments: {e}")