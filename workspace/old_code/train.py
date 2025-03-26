import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import wandb
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
import json
from typing import Dict, List, Tuple, Optional

from model import TrollDetector
from preprocess_data import TrollTweetDataset, load_and_clean_data, create_data_splits, collate_batch, load_and_clean_machova_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrollDetectorTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        num_epochs: int = 10,
        warmup_steps: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.use_wandb = use_wandb
        
        # Initialize mixed precision training
        self.scaler = GradScaler()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(train_loader) * num_epochs
        )

        # Tracking best model
        self.best_val_auc = 0.0
        self.best_epoch = 0
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(project="troll-detection", entity="openhands")
            wandb.watch(self.model)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tweets_per_account=self.train_loader.dataset.tweets_per_account
                )
                loss = self.criterion(outputs['logits'], labels)
            
            total_loss += loss.item()
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Clip gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Update weights with gradient scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Store predictions and labels
            preds = torch.argmax(outputs['logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Calculate metrics
        metrics = self.calculate_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on the given dataloader"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tweets_per_account=dataloader.dataset.tweets_per_account
            )
            
            loss = self.criterion(outputs['logits'], labels)
            total_loss += loss.item()
            
            probs = torch.softmax(outputs['logits'], dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        metrics = self.calculate_metrics(all_preds, all_labels, all_probs)
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics

    def calculate_metrics(
        self,
        preds: List[int],
        labels: List[int],
        probs: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """Calculate classification metrics"""
        report = classification_report(labels, preds, output_dict=True)
        metrics = {
            'accuracy': report['accuracy'],
            'f1': report['weighted avg']['f1-score'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall']
        }
        
        if probs is not None:
            metrics['auc'] = roc_auc_score(labels, probs)
        
        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest_checkpoint.pt')
        
        # Save best model if this is the best so far
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pt')
            
            # Save configuration and metrics
            config = {
                'model_name': self.model.__class__.__name__,
                'best_epoch': epoch,
                'best_metrics': metrics
            }
            with open(self.checkpoint_dir / 'best_model_info.json', 'w') as f:
                json.dump(config, f, indent=4)

    def train(self) -> Dict[str, float]:
        """Main training loop"""
        logger.info(f"Starting training on device: {self.device}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Training phase
            train_metrics = self.train_epoch()
            logger.info(f"Training metrics: {train_metrics}")
            
            # Validation phase
            val_metrics = self.evaluate(self.val_loader)
            logger.info(f"Validation metrics: {val_metrics}")
            
            # Log metrics
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **{f'train_{k}': v for k, v in train_metrics.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                })
            
            # Save checkpoint if this is the best model
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.best_epoch = epoch
                logger.info(f"New best model with validation AUC: {self.best_val_auc:.4f}")
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            
            # Regular checkpoint saving
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, val_metrics)
        
        # Final evaluation on test set if available
        if self.test_loader is not None:
            logger.info("\nEvaluating on test set...")
            test_metrics = self.evaluate(self.test_loader)
            logger.info(f"Test metrics: {test_metrics}")
            
            if self.use_wandb:
                wandb.log({f'test_{k}': v for k, v in test_metrics.items()})
            
            return test_metrics
        
        return val_metrics

def check_dataset_balance(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    def print_class_distribution(df: pd.DataFrame, dataset_name: str):
        class_counts = df['troll'].value_counts()
        total = class_counts.sum()
        print(f"\n{dataset_name} set class distribution:")
        for label, count in class_counts.items():
            print(f"Class {label}: {count} samples ({count / total:.2%})")
    
    print_class_distribution(train_df, "Training")
    print_class_distribution(val_df, "Validation")
    print_class_distribution(test_df, "Test")

def main(data_source: str = 'twitter'):
    """
    Train model on either English tweets or Czech comments.
    
    Args:
        data_source: Either 'twitter' or 'reddit'
    """
    # Training configuration
    config = {
        'model_name': "distilbert-base-multilingual-cased",
        'max_length': 128,
        'tweets_per_account': 10 if data_source == 'english' else 5,  # Fewer for Czech
        'batch_size': 8,
        'learning_rate': 1e-5,
        'num_epochs': 3,
        'warmup_steps': 50,
        'weight_decay': 0.05,
        'max_grad_norm': 0.5,
        'use_wandb': False,
        'dropout_rate': 0.2
    }

    # Load and preprocess data
    data_dir = "./data"
    if data_source == 'twitter':
        all_data = load_and_clean_data(data_dir)
        model_prefix = 'twitter'
    else:
        all_data = load_and_clean_machova_data(data_dir)
        model_prefix = 'reddit'

    train_df, val_df, test_df = create_data_splits(all_data)
    
    # Check dataset balance
    check_dataset_balance(train_df, val_df, test_df)

    # Create datasets and dataloaders
    # ... existing dataset and dataloader creation code ...

    # Initialize model with language-specific checkpoint directory
    checkpoint_dir = f"checkpoints_{model_prefix}"
    model = TrollDetector(model_name=config['model_name'])

    # Initialize trainer
    trainer = TrollDetectorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        warmup_steps=config['warmup_steps'],
        weight_decay=config['weight_decay'],
        max_grad_norm=config['max_grad_norm'],
        use_wandb=config['use_wandb'],
        checkpoint_dir=checkpoint_dir
    )

    # Train model
    final_metrics = trainer.train()
    logger.info(f"Training completed. Final metrics: {final_metrics}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=['twitter', 'reddit'], default='twitter',
                       help='Which dataset to use for training')
    args = parser.parse_args()
    
    main(data_source=args.data)