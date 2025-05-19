import pandas as pd
from pathlib import Path
import json
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class CzechBenchmark:
    """Benchmark dataset for testing model performance on Czech data"""
    
    def __init__(self, benchmark_path: str = "../output/benchmark"):
        self.benchmark_path = Path(benchmark_path)
        self.benchmark_path.mkdir(parents=True, exist_ok=True)
        
        # Default paths
        self.data_file = self.benchmark_path / "benchmark_cases.json"
        self.results_file = self.benchmark_path / "benchmark_results.json"
        
    def create_benchmark_set(self, 
                           comments_df: pd.DataFrame,
                           n_authors: int = 10,
                           min_comments: int = 5) -> None:
        """Create a benchmark set from Czech comments data"""
        
        # Get authors with minimum required comments
        author_counts = comments_df['author'].value_counts()
        eligible_authors = author_counts[author_counts >= min_comments].index[:n_authors]
        
        benchmark_data = {}
        for author in eligible_authors:
            author_comments = comments_df[comments_df['author'] == author]
            benchmark_data[author] = {
                'comments': author_comments['text'].tolist(),
                'article_urls': author_comments['url'].tolist(),
                'timestamps': author_comments['timestamp'].astype(str).tolist()
            }
        
        # Save benchmark data
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Created benchmark set with {len(benchmark_data)} authors")
        
    def run_benchmark(self, predictor) -> Dict:
        """Run benchmark tests using the provided predictor"""
        
        # Load benchmark data
        with open(self.data_file, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
            
        results = {
            'predictions': {},
            'summary': {
                'total_authors': len(benchmark_data),
                'total_comments': sum(len(data['comments']) for data in benchmark_data.values()),
                'troll_predictions': 0
            }
        }
        
        # Run predictions for each author
        for author, data in benchmark_data.items():
            pred = predictor.predict(data['comments'])
            results['predictions'][author] = {
                'prediction': pred['prediction'],
                'confidence': float(pred['trolliness_score']),  # Use trolliness score directly
            }
            
            if pred['prediction'] == 'troll':
                results['summary']['troll_predictions'] += 1
        
        # Save results
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        result_file = self.benchmark_path / f"benchmark_results_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        return results
    
    def add_authors(self, authors: List[str], comments_df: pd.DataFrame) -> None:
        """
        Add specific authors to the benchmark set
        
        Args:
            authors: List of author names to add
            comments_df: DataFrame containing comments data
        """
        # Load existing benchmark data if it exists
        benchmark_data = {}
        if self.data_file.exists():
            with open(self.data_file, 'r', encoding='utf-8') as f:
                benchmark_data = json.load(f)
        
        # Add each author
        for author in authors:
            author_comments = comments_df[comments_df['author'] == author]
            if len(author_comments) == 0:
                logger.warning(f"No comments found for author: {author}")
                continue
                
            benchmark_data[author] = {
                'comments': author_comments['text'].tolist(),
                'article_urls': author_comments['url'].tolist(),
                'timestamps': author_comments['timestamp'].astype(str).tolist()
            }
            logger.info(f"Added {len(author_comments)} comments from {author}")
        
        # Save updated benchmark data
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Benchmark set now contains {len(benchmark_data)} authors")