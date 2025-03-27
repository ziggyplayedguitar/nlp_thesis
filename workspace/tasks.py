from pathlib import Path
import pandas as pd
import torch
from src.data_tools.preprocessor import load_and_clean_data
from src.data_tools.dataset import create_data_splits, TrollDataset, collate_batch
from src.models.bert_model import TrollDetector
from src.models.trainer import TrollDetectorTrainer
from torch.utils.data import DataLoader
import logging
from typing import Dict, Any
import json

from src.data_tools.czech_data_tools import load_czech_media_data
from src.models.predictor import TrollPredictor
from src.data_tools.benchmark import CzechBenchmark

def preprocess(data_dir: str = '../data', 
              output_dir: str = '../data/processed',
              train_size: float = 0.7,
              val_size: float = 0.15,
              random_state: int = 42):
    """Load and preprocess data, create train/val/test splits"""
    # Load and clean data
    df_raw = load_and_clean_data(data_dir)
    
    # Create splits
    train_df, val_df, test_df = create_data_splits(
        df_raw,
        train_size=train_size,
        val_size=val_size,
        random_state=random_state
    )
    
    # Save splits
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_df in [
        ('train', train_df),
        ('val', val_df),
        ('test', test_df)
    ]:
        split_df.to_parquet(output_path / f'{split_name}.parquet')
    
    return {
        'train_path': str(output_path / 'train.parquet'),
        'val_path': str(output_path / 'val.parquet'),
        'test_path': str(output_path / 'test.parquet')
    }

def train(
    train_path: str,
    val_path: str,
    test_path: str,
    model_dir: str = '../checkpoints',
    config: dict = None
):
    """Train model using preprocessed data"""
    if config is None:
        config = {
            'model_name': 'distilbert-base-multilingual-cased',
            'max_length': 64,
            'batch_size': 64,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'num_epochs': 3,
            'warmup_steps': 0,
            'max_grad_norm': 1.0,
            'comments_per_user': 5,
            'use_wandb': False
        }
    
    # Load data
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    # Create datasets and dataloaders
    train_dataset = TrollDataset(
        train_df,
        tokenizer_name=config['model_name'],
        max_length=config['max_length'],
        comments_per_user=config['comments_per_user']
    )
    
    val_dataset = TrollDataset(
        val_df,
        tokenizer_name=config['model_name'],
        max_length=config['max_length'],
        comments_per_user=config['comments_per_user']
    )
    
    test_dataset = TrollDataset(
        test_df,
        tokenizer_name=config['model_name'],
        max_length=config['max_length'],
        comments_per_user=config['comments_per_user']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_batch
    )
    
    # Initialize model and trainer
    model = TrollDetector(model_name=config['model_name'])
    
    trainer = TrollDetectorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        max_grad_norm=config['max_grad_norm'],
        num_epochs=config['num_epochs'],
        warmup_steps=config['warmup_steps'],
        checkpoint_dir=model_dir,
        use_wandb=config['use_wandb']
    )
    
    # Train model
    metrics = trainer.train()
    
    return {
        'model_path': str(Path(model_dir) / 'best_model.pt'),
        'metrics_path': str(Path(model_dir) / 'best_model_info.json')
    }

def run_benchmark(
    model_path: str,
    benchmark_dir: str = '../data/benchmark',
    config: dict = None
) -> Dict[str, str]:
    """Run benchmark tests on Czech data using trained model
    
    Args:
        model_path: Path to trained model checkpoint
        benchmark_dir: Directory for benchmark data and results
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with paths to benchmark outputs
    """
    if config is None:
        config = {
            'comments_per_user': 5,
            'max_length': 64,
            'manual_authors': [
                "Pavel Hanzl",
                "Jitka Bártová",
                "Roman Myška"
            ]
        }
    
    # Load Czech data
    czech_comments = load_czech_media_data()
    
    # Initialize benchmark
    benchmark = CzechBenchmark(benchmark_path=benchmark_dir)
    
    # Create or update benchmark set
    if not benchmark.data_file.exists():
        # Create automatic benchmark set
        benchmark.create_benchmark_set(
            comments_df=czech_comments,
            n_authors=10,
            min_comments=config['comments_per_user']
        )
        # Add manual authors
        benchmark.add_authors(config['manual_authors'], czech_comments)
    
    # Initialize predictor
    predictor = TrollPredictor(
        model_path=model_path,
        comments_per_user=config['comments_per_user'],
        max_length=config['max_length']
    )
    
    # Run benchmark
    results = benchmark.run_benchmark(predictor)
    
    # Save latest results
    benchmark_path = Path(benchmark_dir)
    latest_results = benchmark_path / "latest_results.json"
    history_file = benchmark_path / "history.json"
    
    with open(latest_results, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Update history
    if history_file.exists():
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    else:
        history = []
    
    # Add current results to history
    history.append({
        'timestamp': pd.Timestamp.now().strftime('%Y%m%d_%H%M%S'),
        'model_path': model_path,
        'results': results
    })
    
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    return {
        'benchmark_data': str(benchmark.data_file),
        'latest_results': str(latest_results),
        'history': str(history_file)
    }

