"""Trainer class."""

import logging
import os
from transformers import AutoTokenizer

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from utils.train_utils import log_result, write_to_tensorboard, write_to_csv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for training the model."""

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        total_epochs: int,
        total_steps: int,
        lr_scheduler: optim.lr_scheduler,
        device: str,
        checkpoint_dir: str = None,
        tensorboard_dir: str = None,
        result_dir: str = None,
        gradient_clip_value: float = None,
        steps_to_accumulate: int = 1,
    ):
        """Initialize the Trainer class."""
        # Data Loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Model
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        # Hyperparameters
        self.total_epochs = total_epochs
        self.total_steps = total_steps
        self.gradient_clip_value = gradient_clip_value
        self.steps_to_accumulate = steps_to_accumulate
        # Other
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_dir = tensorboard_dir
        self.result_dir = result_dir

        self.sigmoid = nn.Sigmoid()
        self.model.to(self.device)

        # Tensorboard
        self.writer = None
        if self.tensorboard_dir:
            self.writer = SummaryWriter(self.tensorboard_dir)

    def training(self):
        """Run complete training and validation process."""
        logger.info(
            f"Starting training for {self.total_epochs} epochs on {self.device}\n"
        )

        # Initial validation and log result
        logger.info("Initial validation before training...")
        with torch.no_grad():
            init_epoch_loss = self._epoch(is_training=False, epoch=0)

        self._visulaize(epoch=0, epoch_loss=init_epoch_loss, mode="Validation")

        for epoch in range(1, self.total_epochs + 1):
            logger.info(f"Epoch [{epoch}]")
            # Training phase
            train_epoch_loss, train_epoch_metrics = self._epoch(
                is_training=True, epoch=epoch
            )

            # Validation phase
            with torch.no_grad():
                val_epoch_loss, val_epoch_metrics = self._epoch(
                    is_training=False, epoch=epoch
                )

            self._visulaize(
                epoch=epoch,
                epoch_loss=train_epoch_loss,
                epoch_metrics=train_epoch_metrics,
                mode="Train",
            )
            self._visulaize(
                epoch=epoch,
                epoch_loss=val_epoch_loss,
                epoch_metrics=val_epoch_metrics,
                mode="Validation",
            )

            # Update learning rate
            self.lr_scheduler.step()

            # Save model checkpoint
            if self.checkpoint_dir:
                self._save_checkpoint(epoch=epoch)

            if val_epoch_loss["total_loss"] < self.best_val_loss:
                self.best_val_loss = val_epoch_loss["total_loss"]
                self.best_epoch = epoch
                self.best_model = self.model

        if self.checkpoint_dir:
            self._save_final_model()
            self._save_best_model()
        if self.tensorboard_dir:
            self.writer.close()
        return

    def _epoch(self, is_training: bool = True, epoch: int = 0):
        """Perform one epoch of training or validation."""
        # set model mode
        self.model.train() if is_training else self.model.eval()

        # Initialize accumulators
        epoch_loss = {"BCE_loss": 0.0}

        loader = self.train_loader if is_training else self.val_loader
        # sample size
        total_batches = len(loader)
        desc = "Train" if is_training else "Validation"

        progress_bar = tqdm(loader, desc=desc, leave=True)

        for batch_idx, batch_data in enumerate(progress_bar):
            cur_step = batch_idx + (epoch - 1) * total_batches
            # Get data
            input_ids, attention_mask, labels = self._get_data(batch_data=batch_data)
            input_ids, attention_mask, labels = (
                input_ids.to(self.device),
                attention_mask.to(self.device),
                labels.to(self.device),
            )

            # Forward Pass
            logit = self.model(source=input_ids, target=labels)

            # Calculate loss
            loss = self.criterion(logit, labels)  # Batch loss
            epoch_loss += loss

            # Backward Pass (training only)
            if is_training:
                self.optimizer.zero_grad()
                loss.backward()
                total_grad_norm = self._calculate_grad_norm()
                self.optimizer.step()
            # if is_training:
            #     if cur_step % self.steps_to_accumulate == 0:
            #         self.optimizer.zero_grad()

            #     total_loss.backward()

            #     if (cur_step + 1) % self.steps_to_accumulate == 0:
            #         # Gradient clipping if specified
            #         if self.gradient_clip_value:
            #             nn.utils.clip_grad_norm_(
            #                 self.model.parameters(), self.gradient_clip_value
            #             )
            #         self.optimizer.step()

            mode = "Train(step)" if is_training else "Validation(Step)"
            write_to_tensorboard(
                epoch=cur_step,
                epoch_loss=loss,
                grad_norm=total_grad_norm,
                writer=self.writer,
                mode=mode,
            )

            # Update progress bar with current batch loss
            progress_bar.set_postfix(
                {
                    "loss": f"{epoch_loss.item():.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        # Update epoch loss
        for loss_key in epoch_loss:
            epoch_loss[loss_key] /= total_batches

        return epoch_loss

    def _calculate_grad_norm(self, norm_type=2):
        """Calculate gradient norm manually.

        Args:
            model: PyTorch model
            norm_type: Type of norm (1, 2, or 'inf')
        """
        total_norm = 0.0

        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type

        total_norm = total_norm ** (1.0 / norm_type)
        return total_norm

    def _get_data(self, batch_data) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract input and target data from batch."""
        input_ids = batch_data["input_ids"]
        attention_mask = batch_data["attention_mask"]
        labels = batch_data["labels"]
        return input_ids, attention_mask, labels

    def _visulaize(self, epoch, epoch_loss, epoch_metrics, mode):
        # Get learning rate
        lr = self.lr_scheduler.get_last_lr()[0]
        log_result(epoch_loss=epoch_loss, epoch_metrics=epoch_metrics, mode=mode, lr=lr)

        if self.tensorboard_dir:
            write_to_tensorboard(
                epoch=epoch,
                epoch_loss=epoch_loss,
                epoch_metrics=epoch_metrics,
                writer=self.writer,
                mode=mode,
                lr=lr,
            )
        if self.result_dir:
            write_to_csv(
                epoch=epoch,
                epoch_loss=epoch_loss,
                epoch_metrics=epoch_metrics,
                result_dir=self.result_dir,
                mode=mode,
                lr=lr,
            )

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_{epoch}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return

    def _save_final_model(self) -> None:
        """Save the final trained model."""
        model_path = os.path.join(self.checkpoint_dir, "final_model.pth")
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Final model saved: {model_path}")
        return

    def _save_best_model(self) -> None:
        """Save the best trained model."""
        model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        torch.save(self.best_model.state_dict(), model_path)
        logger.info(f"Best model (Epoch {self.best_epoch}) saved: {model_path}")
        return
