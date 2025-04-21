import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple

def analyze_hourly_patterns(comments_df: pd.DataFrame, min_comments: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Analyze posting time patterns for users.
    
    Args:
        comments_df: DataFrame with comments
        min_comments: Minimum number of comments for a user to be analyzed
    
    Returns:
        Dictionary with user statistics about posting times
    """
    # Create a copy of the dataframe to avoid warnings
    df = comments_df.copy()
    
    # Group comments by user and hour
    df.loc[:, 'hour'] = df['timestamp'].dt.hour
    hourly_activity = df.groupby(['author', 'hour']).size().reset_index(name='count')
    
    # Calculate average activity pattern across all users
    avg_hourly = hourly_activity.groupby('hour')['count'].mean()
    
    # Find users with unusual posting times
    user_hour_stats = {}
    for user in df['author'].unique():
        user_comments = df[df['author'] == user]
        if len(user_comments) >= min_comments:
            # Calculate percentage of posts during "unusual" hours (e.g., 1AM-5AM)
            unusual_hours = set(range(1, 5))
            unusual_posts = user_comments[user_comments['hour'].isin(unusual_hours)]
            unusual_ratio = len(unusual_posts) / len(user_comments)
            
            user_hour_stats[user] = {
                'total_comments': len(user_comments),
                'unusual_hour_ratio': unusual_ratio,
                'peak_hour': user_comments['hour'].mode().iloc[0]
            }
    
    return user_hour_stats, avg_hourly

def find_activity_bursts(user_comments: pd.DataFrame, time_window_minutes: int = 5, min_burst_size: int = 3) -> List[Dict[str, Any]]:
    """
    Find bursts of activity in user comments.
    
    Args:
        user_comments: DataFrame with comments from a single user
        time_window_minutes: Time window to consider for bursts
        min_burst_size: Minimum number of comments to consider a burst
    
    Returns:
        List of dictionaries containing burst information
    """
    if len(user_comments) < min_burst_size:
        return []
    
    # Create a copy and sort comments by timestamp
    df = user_comments.copy().sort_values('timestamp')
    
    # Find time differences between consecutive comments
    time_diffs = df['timestamp'].diff()
    
    # Identify bursts (many comments in short time)
    burst_starts = time_diffs[time_diffs <= timedelta(minutes=time_window_minutes)].index
    
    # Group consecutive burst comments
    bursts = []
    current_burst = []
    
    for i in range(len(df)):
        if i in burst_starts:
            current_burst.append(i)
        elif current_burst:
            if len(current_burst) >= min_burst_size:
                burst_comments = df.iloc[current_burst]
                bursts.append({
                    'start_time': burst_comments.iloc[0]['timestamp'],
                    'end_time': burst_comments.iloc[-1]['timestamp'],
                    'num_comments': len(burst_comments),
                    'comments': burst_comments
                })
            current_burst = []
    
    return bursts

def analyze_burst_patterns(comments_df: pd.DataFrame, min_comments: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Analyze burst patterns for all users.
    
    Args:
        comments_df: DataFrame with comments
        min_comments: Minimum number of comments for a user to be analyzed
    
    Returns:
        Dictionary with user statistics about burst patterns
    """
    user_burst_stats = {}
    for user in comments_df['author'].unique():
        user_comments = comments_df[comments_df['author'] == user]
        bursts = find_activity_bursts(user_comments)
        
        if bursts:
            user_burst_stats[user] = {
                'num_bursts': len(bursts),
                'avg_burst_size': np.mean([b['num_comments'] for b in bursts]),
                'max_burst_size': max(b['num_comments'] for b in bursts),
                'bursts': bursts
            }
    
    return user_burst_stats

def find_coordinated_activity(comments_df: pd.DataFrame, time_window_minutes: int = 15) -> Dict[str, List[Dict[str, Any]]]:
    """
    Find instances of coordinated activity between users.
    
    Args:
        comments_df: DataFrame with comments
        time_window_minutes: Time window to consider for coordination
    
    Returns:
        Dictionary mapping users to their coordinated activity events
    """
    # Create a copy of the dataframe to avoid warnings
    df = comments_df.copy()
    
    # Add debug prints
    print(f"\nDebug: Time window minutes: {time_window_minutes}")
    print(f"Debug: Sample of timestamps before flooring:")
    print(df['timestamp'].head())
    
    # Group comments by article and time window
    df.loc[:, 'time_window'] = df['timestamp'].dt.floor(f'{time_window_minutes}min')
    
    print(f"\nDebug: Sample of time windows after flooring:")
    print(df['time_window'].head())
    
    # Find instances where multiple users comment on the same articles in the same time window
    coordination = defaultdict(list)
    
    # Debug counter for coordinated groups
    total_groups = 0
    
    for (article_id, time_window), group in df.groupby(['article_id', 'time_window']):
        unique_authors = len(group['author'].unique())
        if unique_authors >= 3:  # At least 3 users commenting
            total_groups += 1
            authors = group['author'].unique()
            for author in authors:
                coordination[author].append({
                    'article_id': article_id,
                    'time_window': time_window,
                    'co_commenters': [a for a in authors if a != author]
                })
    
    print(f"\nDebug: Found {total_groups} coordinated groups")
    print(f"Debug: Found {len(coordination)} users with any coordination")
    
    return coordination

def calculate_user_scores(user_hour_stats: Dict[str, Dict[str, Any]], 
                        user_burst_stats: Dict[str, Dict[str, Any]], 
                        coordinated_activity: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    """
    Calculate suspicion scores for users based on their activity patterns.
    
    Args:
        user_hour_stats: Dictionary with user hour statistics
        user_burst_stats: Dictionary with user burst statistics
        coordinated_activity: Dictionary with user coordination information
    
    Returns:
        Dictionary mapping users to their suspicion scores
    """
    user_scores = {}
    for user in user_hour_stats.keys():
        score = 0
        
        # Unusual hours score
        score += user_hour_stats[user]['unusual_hour_ratio'] * 5
        
        # Burst activity score
        if user in user_burst_stats:
            score += (user_burst_stats[user]['num_bursts'] / 
                     user_hour_stats[user]['total_comments']) * 3
        
        # Coordination score
        if user in coordinated_activity:
            score += (len(coordinated_activity[user]) / 
                     user_hour_stats[user]['total_comments']) * 2
        
        user_scores[user] = score
    
    return user_scores

def plot_temporal_patterns(avg_hourly: pd.Series, 
                         user_hour_stats: Dict[str, Dict[str, Any]], 
                         user_burst_stats: Dict[str, Dict[str, Any]], 
                         coordinated_activity: Dict[str, List[Dict[str, Any]]]):
    """
    Create visualizations of temporal patterns.
    
    Args:
        avg_hourly: Series with average hourly activity
        user_hour_stats: Dictionary with user hour statistics
        user_burst_stats: Dictionary with user burst statistics
        coordinated_activity: Dictionary with user coordination information
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Average hourly activity pattern
    plt.subplot(2, 2, 1)
    sns.lineplot(x=avg_hourly.index, y=avg_hourly.values)
    plt.title('Average Hourly Activity Pattern')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Number of Comments')
    
    # Plot 2: Distribution of unusual hour ratios
    plt.subplot(2, 2, 2)
    unusual_ratios = [stats['unusual_hour_ratio'] for stats in user_hour_stats.values()]
    sns.histplot(unusual_ratios, bins=30)
    plt.title('Distribution of Unusual Hour Activity')
    plt.xlabel('Ratio of Posts During Unusual Hours')
    plt.ylabel('Number of Users')
    
    # Plot 3: Burst activity patterns
    plt.subplot(2, 2, 3)
    burst_sizes = [stats['max_burst_size'] for stats in user_burst_stats.values()]
    sns.histplot(burst_sizes, bins=30)
    plt.title('Distribution of Maximum Burst Sizes')
    plt.xlabel('Maximum Comments in a Burst')
    plt.ylabel('Number of Users')
    
    # Plot 4: Coordination network size
    plt.subplot(2, 2, 4)
    coordination_sizes = [len(set(sum([event['co_commenters'] for event in events], []))) 
                         for events in coordinated_activity.values()]
    sns.histplot(coordination_sizes, bins=30)
    plt.title('Distribution of Coordination Network Sizes')
    plt.xlabel('Number of Co-commenters')
    plt.ylabel('Number of Users')
    
    plt.tight_layout()
    plt.show()

def analyze_temporal_patterns(comments_df: pd.DataFrame, 
                            min_comments: int = 10,
                            max_users: Optional[int] = None,
                            random_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Analyze temporal patterns in user posting behavior.
    
    Args:
        comments_df: DataFrame with comments
        min_comments: Minimum number of comments for a user to be analyzed
        max_users: Maximum number of users to analyze (for testing)
        random_seed: Random seed for user sampling
    
    Returns:
        Dictionary containing all analysis results
    """
    # Create a copy of the dataframe to avoid warnings
    df = comments_df.copy()
    
    # Sample users if max_users is specified
    if max_users is not None:
        if random_seed is not None:
            np.random.seed(random_seed)
        users = np.random.choice(df['author'].unique(), 
                               size=min(max_users, len(df['author'].unique())), 
                               replace=False)
        df = df[df['author'].isin(users)]
    
    print("Analyzing temporal patterns...")
    
    # Analyze hourly patterns
    user_hour_stats, avg_hourly = analyze_hourly_patterns(df, min_comments)
    
    # Analyze burst patterns
    user_burst_stats = analyze_burst_patterns(df, min_comments)
    
    # Find coordinated activity
    coordinated_activity = find_coordinated_activity(df)
    
    # Calculate user scores
    user_scores = calculate_user_scores(user_hour_stats, user_burst_stats, coordinated_activity)
    
    # Create visualizations
    plot_temporal_patterns(avg_hourly, user_hour_stats, user_burst_stats, coordinated_activity)
    
    # Display top suspicious users
    top_suspicious = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\nMost suspicious users based on temporal patterns:")
    for i, (user, score) in enumerate(top_suspicious, 1):
        print(f"\n{i}. User: {user} (Suspicion Score: {score:.2f})")
        
        if user in user_hour_stats:
            print(f"   - {user_hour_stats[user]['unusual_hour_ratio']*100:.1f}% posts during unusual hours")
            print(f"   - Peak activity hour: {user_hour_stats[user]['peak_hour']}:00")
        
        if user in user_burst_stats:
            print(f"   - {user_burst_stats[user]['num_bursts']} bursts of activity")
            print(f"   - Max burst size: {user_burst_stats[user]['max_burst_size']} comments")
        
        if user in coordinated_activity:
            num_coordinated = len(coordinated_activity[user])
            print(f"   - {num_coordinated} instances of coordinated activity")
    
    return {
        'user_hour_stats': user_hour_stats,
        'user_burst_stats': user_burst_stats,
        'coordinated_activity': coordinated_activity,
        'user_scores': user_scores,
        'avg_hourly': avg_hourly
    }

def analyze_coordination_patterns(comments_df: pd.DataFrame, 
                               temporal_analysis: Dict[str, Any], 
                               time_window_minutes: int = 15, 
                               min_coordinated_events: int = 5,
                               max_users: Optional[int] = None) -> None:
    """
    Analyze the nature of coordinated behavior between users.
    
    Args:
        comments_df: DataFrame with comments
        temporal_analysis: Results from analyze_temporal_patterns
        time_window_minutes: Time window to consider for coordination
        min_coordinated_events: Minimum number of coordinated events to consider
        max_users: Maximum number of users to analyze in detail
    """
    # Instead of using pre-calculated coordination, calculate it with the specified window
    print(f"\nAnalyzing coordination with {time_window_minutes} minute window...")
    
    # Create a copy of the dataframe to avoid warnings
    df = comments_df.copy()
    
    # Group comments by article and time window
    df.loc[:, 'time_window'] = df['timestamp'].dt.floor(f'{time_window_minutes}min')
    
    # Find instances where multiple users comment on the same articles in the same time window
    coordination = defaultdict(list)
    total_groups = 0
    
    for (article_id, time_window), group in df.groupby(['article_id', 'time_window']):
        unique_authors = len(group['author'].unique())
        if unique_authors >= 3:  # At least 3 users commenting
            total_groups += 1
            authors = group['author'].unique()
            for author in authors:
                coordination[author].append({
                    'article_id': article_id,
                    'time_window': time_window,
                    'co_commenters': [a for a in authors if a != author]
                })
    
    print(f"Found {total_groups} coordinated groups")
    print(f"Found {len(coordination)} users with any coordination")
    
    # Filter for users with significant coordination
    significant_coordination = {
        user: events for user, events in coordination.items()
        if len(events) >= min_coordinated_events
    }
    
    print(f"\nAnalyzing coordination patterns for users with {min_coordinated_events}+ coordinated events")
    print(f"Found {len(significant_coordination)} users with significant coordination")
    
    # Limit the number of users to analyze if specified
    users_to_analyze = list(significant_coordination.keys())
    if max_users is not None:
        users_to_analyze = users_to_analyze[:max_users]
    
    for user in users_to_analyze:
        events = significant_coordination[user]
        print(f"\n=== Detailed Analysis for User: {user} ===")
        print(f"Total coordinated events: {len(events)}")
        
        # 1. Find most frequent co-commenters
        co_commenter_counts = defaultdict(int)
        for event in events:
            for co_commenter in event['co_commenters']:
                co_commenter_counts[co_commenter] += 1
                
        print("\nMost frequent co-commenters:")
        for co_commenter, count in sorted(co_commenter_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"- {co_commenter}: {count} shared events")
            
        # 2. Analyze the articles they coordinate on
        coordinated_articles = defaultdict(int)
        for event in events:
            coordinated_articles[event['article_id']] += 1
            
        print("\nMost frequently coordinated articles:")
        for article_id, count in sorted(coordinated_articles.items(), key=lambda x: x[1], reverse=True)[:3]:
            # Get article title
            article_title = comments_df[comments_df['article_id'] == article_id]['title'].iloc[0]
            print(f"- Article: {article_title[:100]}...")
            print(f"  Coordinated events: {count}")
            
            # Get the actual comments for one coordinated event on this article
            article_events = [e for e in events if e['article_id'] == article_id]
            if article_events:
                event = article_events[0]
                time_window = event['time_window']
                
                # Get comments from this user and co-commenters in this time window
                relevant_comments = comments_df[
                    (comments_df['article_id'] == article_id) & 
                    (comments_df['timestamp'].dt.floor(f'{time_window_minutes}min') == time_window) &
                    (comments_df['author'].isin([user] + event['co_commenters']))
                ]
                
                print("\n  Example comment exchange:")
                for _, comment in relevant_comments.sort_values('timestamp').iterrows():
                    print(f"    {comment['author']}: {comment['content'][:100]}...")
        
        # 3. Analyze sentiment patterns in coordinated comments
        coordinated_comments = comments_df[
            (comments_df['author'] == user) & 
            (comments_df['article_id'].isin(coordinated_articles.keys()))
        ]
        
        if 'sentiment' in comments_df.columns:
            sentiments = coordinated_comments['sentiment'].value_counts()
            total = len(coordinated_comments)
            
            print("\nSentiment pattern in coordinated comments:")
            for sentiment, count in sentiments.items():
                print(f"- {sentiment}: {(count/total):.2%}")
                
        # 4. Time pattern of coordination
        coordination_times = pd.Series([e['time_window'].hour for e in events])
        peak_hours = coordination_times.value_counts().head(3)
        
        print("\nPeak coordination hours:")
        for hour, count in peak_hours.items():
            print(f"- Hour {hour:02d}:00: {count} events") 