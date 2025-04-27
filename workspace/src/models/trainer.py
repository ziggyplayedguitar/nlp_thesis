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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
        # Initialize trainer attributes
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

        # Loss function - using Huber Loss for regression (more robust than MSE)
        self.criterion = nn.HuberLoss()

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
        self.best_val_r2 = -float('inf')  # Using R² score instead of AUC
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
            labels = batch['label'].to(self.device).float()  # Convert to float for regression
            authors = batch['author']
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tweets_per_account=self.train_loader.dataset.comments_per_user
                )
                loss = self.criterion(outputs['trolliness_score'].squeeze(), labels)
            
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
            
            # Store results for this batch
            batch_results[f"batch_{batch_idx}"] = {
                'author': authors,
                'pred': outputs['trolliness_score'].squeeze().detach().cpu().numpy(),
                'label': labels.cpu().numpy()
            }
            
            batch_idx += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Aggregate predictions by author
        aggregated = aggregate_author_predictions(batch_results, method='mean')
        
        # Extract aggregated predictions and labels
        all_preds = [data['final_pred'] for data in aggregated.values()]
        all_labels = [data['true_label'] for data in aggregated.values()]
        
        # Calculate metrics
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
            labels = batch['label'].to(self.device).float()
            authors = batch['author']
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tweets_per_account=dataloader.dataset.comments_per_user
            )
            
            loss = self.criterion(outputs['trolliness_score'].squeeze(), labels)
            total_loss += loss.item()
            
            # Store results for this batch
            batch_results[f"batch_{batch_idx}"] = {
                'author': authors,
                'pred': outputs['trolliness_score'].squeeze().cpu().numpy(),
                'label': labels.cpu().numpy()
            }
            
            batch_idx += 1
        
        # Aggregate predictions by author
        aggregated = aggregate_author_predictions(batch_results, method='mean')
        
        # Extract aggregated predictions and labels
        all_preds = [data['final_pred'] for data in aggregated.values()]
        all_labels = [data['true_label'] for data in aggregated.values()]
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / len(dataloader)
        metrics['num_authors'] = len(aggregated)
        
        return metrics

    def calculate_metrics(
        self,
        preds: List[float],
        labels: List[float]
    ) -> Dict[str, float]:
        """Calculate regression metrics"""
        metrics = {
            'mse': mean_squared_error(labels, preds),
            'rmse': np.sqrt(mean_squared_error(labels, preds)),  # Calculate RMSE manually
            'mae': mean_absolute_error(labels, preds),
            'r2': r2_score(labels, preds)
        }
        
        # Add binary classification metrics using 0.5 threshold for comparison
        binary_preds = [1 if p >= 0.5 else 0 for p in preds]
        binary_labels = [1 if l >= 0.5 else 0 for l in labels]
        accuracy = sum(p == l for p, l in zip(binary_preds, binary_labels)) / len(preds)
        metrics['binary_accuracy'] = accuracy
        
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
            if val_metrics['r2'] > self.best_val_r2:
                self.best_val_r2 = val_metrics['r2']
                self.best_epoch = epoch
                logger.info(f"New best model with validation R²: {self.best_val_r2:.4f}")
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