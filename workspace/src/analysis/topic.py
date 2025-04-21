import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_json_in_chunks_with_ids(file_path: str, chunk_size: int = 1000, max_entries: int = 2000):
    """
    Read a large JSON file in chunks, adding article IDs to each entry.
    
    Args:
        file_path: Path to the JSON file
        chunk_size: Number of entries to yield at once
        max_entries: Maximum number of entries to process
        
    Yields:
        Tuple of (articles, comments) where each article has a unique article_id
        and each comment has a reference to its article_id
    """
    articles = []
    comments = []
    article_id_counter = 0
    entry_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read the file line by line
        current_entry = []
        bracket_count = 0
        
        for line in tqdm(f, desc="Reading JSON file"):
            if entry_count >= max_entries:
                break
                
            line = line.strip()
            if not line:
                continue
                
            # Count brackets in the line
            bracket_count += line.count('{') - line.count('}')
            current_entry.append(line)
            
            # If we've closed all brackets, we have a complete entry
            if bracket_count == 0 and current_entry:
                try:
                    entry_str = ''.join(current_entry)
                    # Remove trailing comma if present
                    if entry_str.endswith(','):
                        entry_str = entry_str[:-1]
                    entry = json.loads(entry_str)
                    
                    # Process article
                    article = entry['article']
                    article['article_id'] = article_id_counter
                    articles.append(article)
                    
                    # Process comments
                    for comment in entry['comments']:
                        comment['article_id'] = article_id_counter
                        comments.append(comment)
                    
                    article_id_counter += 1
                    entry_count += 1
                    
                    # Yield chunks if we've reached chunk_size
                    if len(articles) >= chunk_size:
                        yield articles, comments
                        articles = []
                        comments = []
                    
                    current_entry = []
                except json.JSONDecodeError:
                    current_entry = []
                    continue
    
    # Yield any remaining data
    if articles:
        yield articles, comments

def parse_topic_file(topic_file_path: str, articles_df: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
    """
    Parse the topic file and extract cluster information.
    
    Args:
        topic_file_path: Path to the topic file
        articles_df: DataFrame containing articles
        
    Returns:
        Tuple of dictionaries mapping article IDs to clusters, topics, and titles to clusters
    """
    # Initialize structures for article-to-cluster mapping
    article_id_to_cluster = {}
    topic_article_to_cluster = {}  # For debugging
    title_to_cluster = {}  # For title-based lookup

    # Reset cluster columns in articles_df
    articles_df['cluster_id'] = -1
    articles_df['cluster_label'] = 'Unlabelled'

    # Read entire file first to understand structure
    with open(topic_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Process the file structure correctly
    current_cluster = None
    current_label = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        i += 1
        
        if not line or line.startswith('='):
            continue
        
        # New cluster section starts
        if line.startswith('cluster'):
            parts = line.split(':')
            cluster_info = parts[0].split()
            current_cluster = int(cluster_info[1])  # Explicitly convert to int
            current_label = parts[1].strip()
            # logger.info(f"Found cluster: {current_cluster} - {current_label}")
            
        # Article within current cluster
        elif line.startswith('** Article'):
            parts = line.split(':', 1)
            article_part = parts[0].strip()
            topic_article_id = int(article_part.split()[2])
            article_title = parts[1].strip()
            
            # Store mapping for debugging
            topic_article_to_cluster[topic_article_id] = (current_cluster, current_label)
            title_to_cluster[article_title] = (current_cluster, current_label)
            
            # Find this article in our dataset
            matching_rows = articles_df[articles_df['title'] == article_title]
            if not matching_rows.empty:
                our_article_id = matching_rows.iloc[0]['article_id']
                articles_df.loc[matching_rows.index, 'cluster_id'] = current_cluster
                articles_df.loc[matching_rows.index, 'cluster_label'] = current_label
                article_id_to_cluster[our_article_id] = (current_cluster, current_label)
    
    return article_id_to_cluster, topic_article_to_cluster, title_to_cluster

def update_comments_with_clusters(comments_df: pd.DataFrame, articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Update comments with cluster information from their associated articles.
    
    Args:
        comments_df: DataFrame containing comments
        articles_df: DataFrame containing articles with cluster information
        
    Returns:
        Updated comments DataFrame with cluster information
    """
    # Set default cluster values
    comments_df['cluster_id'] = -1
    comments_df['cluster_label'] = 'Unlabelled'

    # Create a mapping of article_id to cluster info from articles_df
    article_cluster_mapping = {}
    for _, row in articles_df.iterrows():
        article_cluster_mapping[row['article_id']] = (row['cluster_id'], row['cluster_label'])

    # Apply to comments
    for idx, row in comments_df.iterrows():
        if row['article_id'] in article_cluster_mapping:
            cluster_id, cluster_label = article_cluster_mapping[row['article_id']]
            comments_df.at[idx, 'cluster_id'] = cluster_id
            comments_df.at[idx, 'cluster_label'] = cluster_label
    
    return comments_df


def analyze_cluster_distribution(comments_df: pd.DataFrame) -> Dict:
    """
    Analyze the distribution of comments across clusters.
    
    Args:
        comments_df: DataFrame containing comments with cluster information
        
    Returns:
        Dictionary with cluster distribution statistics
    """
    total_comments = len(comments_df)
    
    # Calculate cluster distribution
    cluster_distribution = comments_df['cluster_id'].value_counts().sort_index()
    
    # Get cluster labels
    cluster_label_mapping = {}
    for _, row in comments_df[['cluster_id', 'cluster_label']].drop_duplicates().iterrows():
        cluster_label_mapping[row['cluster_id']] = row['cluster_label']
    
    # Ensure -1 is in the mapping
    if -1 not in cluster_label_mapping:
        cluster_label_mapping[-1] = "Unlabelled"
    
    # Build result
    result = {
        'total_comments': total_comments,
        'clusters': {}
    }
    
    for cluster_id, count in cluster_distribution.items():
        label = cluster_label_mapping.get(cluster_id, "Unknown")
        percentage = count / total_comments * 100
        result['clusters'][cluster_id] = {
            'label': label,
            'count': int(count),
            'percentage': percentage
        }
    
    return result

def analyze_author_clusters(comments_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze which clusters each user posts to.
    
    Args:
        comments_df: DataFrame containing comments with cluster information
        
    Returns:
        Tuple of (user_cluster_df, statistics) where:
        - user_cluster_df is a DataFrame with user cluster statistics
        - statistics is a dictionary with overall statistics
    """
    # Group comments by user and find unique clusters per user
    user_clusters = {}

    # Process each comment to build the mapping
    for _, comment in comments_df.iterrows():
        author = comment['author']
        cluster_id = comment['cluster_id']
        cluster_label = comment['cluster_label']
        
        # Initialize user entry if not exists
        if author not in user_clusters:
            user_clusters[author] = {
                'clusters': set(),  # Use a set to store unique clusters
                'cluster_details': {},  # Map clusters to their labels
                'comment_count': 0
            }
        
        # Add cluster to user's set
        user_clusters[author]['clusters'].add(cluster_id)
        user_clusters[author]['cluster_details'][cluster_id] = cluster_label
        user_clusters[author]['comment_count'] += 1

    # Create DataFrame for analysis
    user_data = []
    for author, data in user_clusters.items():
        user_data.append({
            'author': author,
            'unique_clusters': len(data['clusters']),
            'clusters': list(data['clusters']),
            'cluster_labels': [data['cluster_details'][c] for c in data['clusters']],
            'comment_count': data['comment_count']
        })

    user_cluster_df = pd.DataFrame(user_data)

    # Sort by number of unique clusters (descending)
    user_cluster_df = user_cluster_df.sort_values(by='unique_clusters', ascending=False)

    # Calculate statistics
    total_users = len(user_cluster_df)
    cluster_count_distribution = user_cluster_df['unique_clusters'].value_counts().sort_index()
    
    # Build statistics dictionary
    statistics = {
        'total_users': total_users,
        'distribution': {},
        'top_users': []
    }
    
    for count, num_users in cluster_count_distribution.items():
        percentage = (num_users / total_users) * 100
        statistics['distribution'][int(count)] = {
            'num_users': int(num_users),
            'percentage': percentage
        }
    
    # Add top 10 users
    top_users = user_cluster_df.head(10)
    for _, row in top_users.iterrows():
        top_clusters = []
        clusters_to_show = min(5, len(row['clusters']))
        
        for i in range(clusters_to_show):
            cluster_id = row['clusters'][i]
            label = row['cluster_labels'][i]
            top_clusters.append({
                'cluster_id': int(cluster_id),
                'label': label
            })
        
        statistics['top_users'].append({
            'author': row['author'],
            'unique_clusters': int(row['unique_clusters']),
            'comment_count': int(row['comment_count']),
            'top_clusters': top_clusters
        })
    
    return user_cluster_df, statistics

def calculate_user_sentiment_ratios(comments_df: pd.DataFrame, max_users: int = 100, 
                                   random_sample: bool = True) -> pd.DataFrame:
    """
    Calculate sentiment ratios for a subset of users.
    
    Args:
        comments_df: DataFrame containing comments with attributes column containing sentiment
        max_users: Maximum number of users to process
        random_sample: If True, take a random sample; if False, take first max_users
        
    Returns:
        DataFrame with sentiment ratios for each user
    """
    # Get unique users
    users = comments_df['author'].unique()
    
    # Take a subset of users
    if len(users) > max_users:
        if random_sample:
            import random
            random.seed(42)
            users = random.sample(list(users), max_users)
        else:
            users = users[:max_users]
    
    logger.info(f"Processing {len(users)} users for sentiment analysis...")
    
    # Initialize results dictionary
    results = {
        'author': [],
        'negative_ratio': [],
        'positive_ratio': [],
        'ambivalent_ratio': [],
        'neutral_ratio': [],
        'comment_count': []
    }
    
    # Calculate sentiment ratios for each user
    for user in users:
        user_comments = comments_df[comments_df['author'] == user]
        
        if len(user_comments) == 0:
            continue
        
        # Extract sentiment values
        sentiments = []
        for attrs in user_comments['attributes']:
            if isinstance(attrs, dict) and 'sentiment' in attrs:
                sentiments.append(attrs['sentiment'])
        
        total_comments = len(sentiments)
        if total_comments == 0:
            continue
            
        negative_count = sentiments.count('Negative')
        positive_count = sentiments.count('Positive')
        ambivalent_count = sentiments.count('Ambivalent')
        neutral_count = sentiments.count('Neutral')
        
        # Add to results
        results['author'].append(user)
        results['negative_ratio'].append(negative_count / total_comments if total_comments > 0 else 0)
        results['positive_ratio'].append(positive_count / total_comments if total_comments > 0 else 0)
        results['ambivalent_ratio'].append(ambivalent_count / total_comments if total_comments > 0 else 0)
        results['neutral_ratio'].append(neutral_count / total_comments if total_comments > 0 else 0)
        results['comment_count'].append(total_comments)
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def load_and_process_data(json_file_path: str, topic_file_path: str, 
                          
                          
                         max_entries: int = 200000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and process data from JSON and topic files.
    
    Args:
        json_file_path: Path to the JSON file containing articles and comments
        topic_file_path: Path to the topic file
        max_entries: Maximum number of entries to process from the JSON file
        
    Returns:
        Tuple of (articles_df, comments_df) with processed data
    """
    # Read data from JSON file
    all_articles = []
    all_comments = []

    for article_chunk, comment_chunk in read_json_in_chunks_with_ids(json_file_path, max_entries=max_entries):
        all_articles.extend(article_chunk)
        all_comments.extend(comment_chunk)

    # Create DataFrames
    articles_df = pd.DataFrame(all_articles)
    comments_df = pd.DataFrame(all_comments)

    logger.info(f"Total number of articles: {len(articles_df)}")
    logger.info(f"Total number of comments: {len(comments_df)}")
    
    # Parse topic file and update articles with cluster information
    article_id_to_cluster, _, _ = parse_topic_file(topic_file_path, articles_df)
    
    # Update comments with cluster information
    comments_df = update_comments_with_clusters(comments_df, articles_df)
    
    return articles_df, comments_df 

def search_comments(comments_df, articles_df, search_term, case_sensitive=False, max_comments=None):
    """
    Search for comments containing a specific term.
    
    Args:
        comments_df: DataFrame containing comments
        articles_df: DataFrame containing articles  
        search_term: Term to search for
        case_sensitive: Whether the search should be case-sensitive
        max_comments: Maximum number of comments to check (None = check all)
        
    Returns:
        List of dictionaries with information about matching comments
    """
    import re
    from tqdm.notebook import tqdm
    
    print(f"Searching for comments containing '{search_term}'...")
    
    # Create a regular expression pattern for the search
    if case_sensitive:
        pattern = re.compile(r'\b' + search_term + r'\b')
    else:
        pattern = re.compile(r'\b' + search_term + r'\b', re.IGNORECASE)
    
    # Limit comments if specified
    comments_to_check = comments_df.head(max_comments) if max_comments else comments_df
    print(f"Checking {len(comments_to_check)} out of {len(comments_df)} total comments")
    
    # Initialize a list to store matching comments
    matching_comments = []
    
    # Search through comments
    for _, comment in tqdm(comments_to_check.iterrows(), total=len(comments_to_check)):
        # Check if the comment content contains the search term
        if 'content' in comment and isinstance(comment['content'], str):
            if pattern.search(comment['content']):
                # Get article title
                article_title = "Unknown"
                try:
                    article_title = articles_df[articles_df['article_id'] == comment['article_id']]['title'].values[0]
                except (IndexError, KeyError):
                    pass
                
                # Add to matching comments
                matching_comments.append({
                    'author': comment['author'],
                    'cluster_id': comment['cluster_id'],
                    'cluster_label': comment['cluster_label'],
                    'article_title': article_title,
                    'content': comment['content'],
                    'article_id': comment['article_id']
                })
    
    print(f"Found {len(matching_comments)} comments containing '{search_term}'")
    return matching_comments

# Function to display search results
def display_search_results(matching_comments, max_to_show=20, line_width=80):
    """
    Display the results of a comment search.
    
    Args:
        matching_comments: List of matching comments from search_comments
        max_to_show: Maximum number of comments to display
        line_width: Width for text wrapping
    """
    import textwrap
    
    # Print comment details
    for i, comment in enumerate(matching_comments[:max_to_show]):
        print(f"\n--- Comment {i+1} ---")
        print(f"Author: {comment['author']}")
        print(f"Cluster: {comment['cluster_id']} ({comment['cluster_label']})")
        print(f"Article: {comment['article_title']}")
        print("Content:")
        
        # Format the content with proper line wrapping
        wrapped_content = textwrap.fill(comment['content'], width=line_width)
        print(wrapped_content)
        
    # Show if there are more results not displayed
    if len(matching_comments) > max_to_show:
        print(f"\n... and {len(matching_comments) - max_to_show} more matching comments")
    
    # Print author statistics
    authors = {}
    for comment in matching_comments:
        author = comment['author']
        if author in authors:
            authors[author] += 1
        else:
            authors[author] = 1

    print("\n--- Top authors mentioning the search term ---")
    for author, count in sorted(authors.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{author}: {count} mentions")
