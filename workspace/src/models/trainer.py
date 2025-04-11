import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import logging
import wandb
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
import json
from typing import Dict, List, Tuple, Optional
from src.data_tools.dataset import aggregate_author_predictions

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
        # Initialize trainer attributes (same as in train.py)
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
            wandb.init(project="troll-detection")
            wandb.watch(self.model)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        batch_results = {}
        batch_idx = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            authors = batch['author']  # List of author IDs
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tweets_per_account=self.train_loader.dataset.comments_per_user
                )
                loss = self.criterion(outputs['logits'], labels)
            
            # Rest of the training loop (same as in train.py)
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
            
            # Get predictions
            probs = torch.softmax(outputs['logits'], dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Store results for this batch
            batch_results[f"batch_{batch_idx}"] = {
                'author': authors,
                'pred': preds.cpu().detach().numpy(),
                'probs': probs.cpu().detach().numpy(),
                'label': labels.cpu().detach().numpy()
            }
            
            batch_idx += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Aggregate predictions by author
        aggregated = aggregate_author_predictions(batch_results, method='mean')
        
        # Extract aggregated predictions and labels
        all_preds = [data['final_pred'] for data in aggregated.values()]
        all_labels = [data['true_label'] for data in aggregated.values()]
        
        # Calculate metrics based on aggregated predictions
        metrics = self.calculate_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / len(self.train_loader)
        metrics['num_authors'] = len(aggregated)
        
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on the given dataloader"""
        self.model.eval()
        total_loss = 0
        batch_results = {}
        batch_idx = 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            authors = batch['author']  # List of author IDs
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tweets_per_account=dataloader.dataset.comments_per_user
            )
            
            loss = self.criterion(outputs['logits'], labels)
            total_loss += loss.item()
            
            probs = torch.softmax(outputs['logits'], dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Store results for this batch
            batch_results[f"batch_{batch_idx}"] = {
                'author': authors,
                'pred': preds.cpu().detach().numpy(),
                'probs': probs.cpu().detach().numpy(),
                'label': labels.cpu().detach().numpy()
            }
            
            batch_idx += 1
        
        # Aggregate predictions by author
        aggregated = aggregate_author_predictions(batch_results, method='mean')
        
        # Extract aggregated predictions and labels
        all_preds = [data['final_pred'] for data in aggregated.values()]
        all_labels = [data['true_label'] for data in aggregated.values()]
        all_probs = [data['final_prob'][1] for data in aggregated.values()]  # Probability of class 1 (troll)
        
        # Calculate metrics based on aggregated predictions
        metrics = self.calculate_metrics(all_preds, all_labels, all_probs)
        metrics['loss'] = total_loss / len(dataloader)
        
        # Add the number of unique authors
        metrics['num_authors'] = len(aggregated)
        
        # Log additional information about multiple samples per author
        multi_sample_authors = [author for author, data in aggregated.items() if data['num_batches'] > 1]
        avg_batches = sum(data['num_batches'] for data in aggregated.values()) / len(aggregated)
        logger.info(f"Evaluated {len(aggregated)} unique authors, {len(multi_sample_authors)} with multiple batches")
        logger.info(f"Average batches per author: {avg_batches:.2f}")
        
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