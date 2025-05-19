import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import logging
import wandb
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from typing import Dict, List, Tuple, Optional, Any
from src.data_tools.dataset import aggregate_author_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Helper class for calculating and tracking metrics."""
    
    @staticmethod
    def calculate_metrics(preds: List[float], labels: List[float]) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mse': mean_squared_error(labels, preds),
            'rmse': np.sqrt(mean_squared_error(labels, preds)),
            'mae': mean_absolute_error(labels, preds),
            'r2': r2_score(labels, preds),
            'binary_accuracy': np.mean((np.array(preds) >= 0.5) == np.array(labels))
        }
    
    @staticmethod
    def update_history(history: Dict[str, Dict[str, List[float]]], 
                      split: str, metrics: Dict[str, float]) -> None:
        """Update metrics history."""
        for metric, value in metrics.items():
            if metric not in history[split]:
                history[split][metric] = []
            history[split][metric].append(value)


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

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

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
        self.best_val_r2 = -float('inf')
        self.best_epoch = 0
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(project="troll-detection")
            wandb.watch(self.model)

        self.current_epoch = 0

        # Initialize training history
        self.history = {
            'train': {metric: [] for metric in ['loss', 'mse', 'rmse', 'mae', 'r2', 'binary_accuracy']},
            'val': {metric: [] for metric in ['loss', 'mse', 'rmse', 'mae', 'r2', 'binary_accuracy']},
            'test': {metric: [] for metric in ['loss', 'mse', 'rmse', 'mae', 'r2', 'binary_accuracy']}
        }

    def _process_batch(self, batch: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        """Process a single batch of data."""
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['label'].to(self.device).float()
        authors = batch['author']
        
        # Forward pass
        if is_training:
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tweets_per_account=self.train_loader.dataset.comments_per_user
                )
                loss = self.criterion(outputs['trolliness_score'].view(-1), labels.view(-1))
        else:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tweets_per_account=self.train_loader.dataset.comments_per_user
                )
                loss = self.criterion(outputs['trolliness_score'].view(-1), labels.view(-1))
        
        return {
            'loss': loss,
            'authors': authors,
            'preds': np.atleast_1d(outputs['trolliness_score'].detach().cpu().numpy()),
            'labels': np.atleast_1d(labels.cpu().numpy())
        }

    def _process_epoch(self, dataloader: DataLoader, is_training: bool = True) -> Dict[str, float]:
        """Process a single epoch of data."""
        self.model.train() if is_training else self.model.eval()
        total_loss = 0
        batch_results = {}
        
        pbar = tqdm(dataloader, desc="Training" if is_training else "Evaluating")
        for batch_idx, batch in enumerate(pbar):
            # Process batch
            results = self._process_batch(batch, is_training)
            
            # Update loss
            total_loss += results['loss'].item()
            
            # Backward pass and optimization if training
            if is_training:
                self.scaler.scale(results['loss']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Store results
            batch_results[f"batch_{batch_idx}"] = {
                'author': results['authors'],
                'pred': results['preds'],
                'label': results['labels']
            }
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{results['loss'].item():.4f}"})
        
        # Aggregate predictions by author
        aggregated = aggregate_author_predictions(batch_results, method='mean')
        
        # Prepare predictions for metric calculation
        final_preds_logits = [data['final_pred'] for data in aggregated.values()]
        true_labels = [data['true_label'] for data in aggregated.values()]

        # Convert logits to probabilities FOR METRIC CALCULATION ONLY
        # Ensure final_preds_logits is not empty before converting
        if final_preds_logits:
            final_preds_probs = torch.sigmoid(torch.tensor(final_preds_logits, dtype=torch.float32)).numpy()
        else:
            final_preds_probs = np.array([]) # Handle empty case if no predictions

        # Calculate metrics using PROBABILITIES
        # Ensure that labels are also in a compatible format (e.g. numpy array) if not already
        metrics = MetricsCalculator.calculate_metrics(
            list(final_preds_probs), # Pass probabilities
            true_labels
        )
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        return self._process_epoch(self.train_loader, is_training=True)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on the given dataloader."""
        return self._process_epoch(dataloader, is_training=False)

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if needed
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with R² score: {metrics['r2']:.4f}")
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.save(str(checkpoint_path))
            if is_best:
                wandb.save(str(best_path))

    def train(self) -> Dict[str, float]:
        """Train the model for the specified number of epochs."""
        logger.info("Starting training...")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train and evaluate
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            
            # Update history
            MetricsCalculator.update_history(self.history, 'train', train_metrics)
            MetricsCalculator.update_history(self.history, 'val', val_metrics)
            
            # Log metrics
            logger.info(f"Train metrics: {train_metrics}")
            logger.info(f"Val metrics: {val_metrics}")
            
            if self.use_wandb:
                wandb.log({
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    'epoch': epoch
                })
            
            # Save checkpoint
            is_best = val_metrics['r2'] > self.best_val_r2
            if is_best:
                self.best_val_r2 = val_metrics['r2']
                self.best_epoch = epoch
            
            self.save_checkpoint(epoch, val_metrics, is_best)
        
        # Evaluate on test set if available
        if self.test_loader is not None:
            test_metrics = self.evaluate(self.test_loader)
            MetricsCalculator.update_history(self.history, 'test', test_metrics)
            logger.info(f"Test metrics: {test_metrics}")
            
            if self.use_wandb:
                wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        
        best_epoch = self.best_epoch  # if available
        print(f"\nBest epoch: {best_epoch + 1}")
        print("Best validation metrics:")
        for metric, values in self.history['val'].items():
            print(f"{metric}: {values[best_epoch]:.4f}")
        

        return self.history