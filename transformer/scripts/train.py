import os
from torch.utils.data import DataLoader
import torch
from torch import optim, nn
import logging
import sys
import time
import yaml
import argparse
from datetime import datetime
from transformers import get_linear_schedule_with_warmup
import ast
from datasets import load_from_disk


current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
greatgrandparent_dir = os.path.dirname(grandparent_dir)

sys.path.append(greatgrandparent_dir)

from utils.utils import get_elapsed_time
from transformer.scripts.trainer import Trainer
from transformer.model.transformer import Transformer
from transformer.datasets.dataset import DatasetPreparation


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Training:
    """Main training orchestrator for Transformer models.

    This class manages the complete training workflow including:
    - Directory setup for checkpoints, tensorboard logs, and results
    - Hyperparameter loading and configuration
    - Dataset loading and dataloader initialization
    - Model, optimizer, scheduler, and loss function setup
    - Training execution via Trainer class"""

    def __init__(
        self,
        config: dict,
        model_type: str,
        training_dataset: str,
        save_results: bool = False,
    ):
        """Initialize the Training orchestrator.

        Args:
            config (dict): Configuration dictionary loaded from YAML
            model_type (str): Model architecture type identifier
            training_dataset (str): Dataset identifier from config
            save_results (bool): Whether to save outputs (default: False for debugging)
        """
        self.config = config
        self.model_type = model_type
        self.training_dataset = training_dataset
        self.save_results = save_results

    def _saving_dir(self) -> None:
        """Setup output directories for checkpoints, logs, and results.

        Creates a timestamped directory structure for organizing training outputs:
        - base_dir/transformer/{model_type}/{training_dataset}/{timestamp}/checkpoints/
        - base_dir/transformer/{model_type}/{training_dataset}/{timestamp}/tensorboard/
        - base_dir/transformer/{model_type}/{training_dataset}/{timestamp}/results/

        If save_results is False, directories are set to None (useful for debugging).


        Returns:
            None
        """
        logger.info("Loading Directories...")
        # Initialize as None (used when save_results=False)
        self.checkpoint_dir, self.tensorboard_dir, self.result_dir = None, None, None
        if self.save_results:
            # Create timestamp for unique run identification
            date_time_str = datetime.now().strftime("%m-%d_%H-%M-%S")

            output_dir: dict = self.config["output_dir"]

            # Build hierarchical directory structure
            base_dir: str = os.path.join(
                output_dir["base_dir"],
                "transformer",
                self.model_type,
                self.training_dataset,
                date_time_str,
            )

            # Define specific output directories
            self.checkpoint_dir = os.path.join(base_dir, output_dir["checkpoint_dir"])
            self.tensorboard_dir = os.path.join(base_dir, output_dir["tensorboard_dir"])
            self.result_dir = os.path.join(base_dir, output_dir["result_dir"])

            # Create all directories
            for directories in [
                self.checkpoint_dir,
                self.tensorboard_dir,
                self.result_dir,
            ]:
                os.makedirs(directories, exist_ok=True)
        logger.info(
            f"Output Directories: \n"
            f"\tCheckpoint Dir: {self.checkpoint_dir}\n"
            f"\tTensorboard Dir: {self.tensorboard_dir}\n"
            f"\tResults Dir: {self.result_dir}\n"
        )

    def _hyperparameters(self) -> None:
        """Load and set all hyperparameters from the configuration.

        Returns:
            None
        """
        logger.info("Loading Hyperparameters...")

        # ==================== Model Architecture ====================
        model_hparams: dict = self.config[self.model_type]
        self.max_seq_len: int = model_hparams["max_seq_len"]  # Max tokens per sequence
        self.d_model: int = model_hparams["d_model"]  # Model/embedding dimension
        self.d_ff: int = model_hparams["d_ff"]  # Feed-forward network dimension
        # Encoder/decoder layer count
        self.num_layers: int = model_hparams["num_layers"]
        self.num_heads: int = model_hparams["num_heads"]  # Multi-head attention heads
        self.dropout: float = model_hparams["dropout"]  #  Dropout rates
        self.total_steps: int = model_hparams["total_steps"]  # Total training steps

        # ==================== Optimizer Configuration ====================
        optimizer_hparams: dict = self.config["optimizer"]
        self.lr: float = optimizer_hparams["lr"]  # Learning rate
        self.betas: tuple[float, float] = ast.literal_eval(optimizer_hparams["betas"])
        self.eps: float = optimizer_hparams["eps"]  # Epsilon for numerical stability

        # ==================== Learning Rate Scheduler ====================
        self.warmup_steps: float = self.config["scheduler"]["warmup_steps"]

        # ==================== Loss Function ====================
        self.label_smoothing: float = self.config["loss"]["label_smoothing"]

        # ==================== Training Configuration ====================
        training_hparams: dict = self.config["training"]
        cuda: int = training_hparams["cuda"]  #  GPU device number
        self.batch_size: int = training_hparams["batch_size"]
        self.total_epochs: int = training_hparams["total_epochs"]

        # Setup device (GPU if available, else CPU)
        self.device = f"{cuda}" if torch.cuda.is_available() else "cpu"

        # ==================== Dataset Configuration ====================
        dataset = self.config[self.training_dataset]
        self.processed_data_dir = dataset["processed_data_dir"]
        self.vocab_size = dataset["vocab_size"]  # Shared vocab size for both languages

    def _load_dataset(self) -> None:
        """Load pre-tokenized datasets from disk.

        Loads training and validation datasets that were previously tokenized
        by the DatasetPreparation pipeline. Expects datasets to be saved in
        HuggingFace datasets format.


        Returns:
            None

        Raises:
            FileNotFoundError: If dataset directories don't exist
        """
        logger.info("Loading Datasets...")
        # Construct paths to tokenized data splits
        train_dir = os.path.join(self.processed_data_dir, "train")
        valid_dir = os.path.join(self.processed_data_dir, "valid")

        # Load the pre-tokenized datasets from disk
        self.train_set = load_from_disk(train_dir)
        self.valid_set = load_from_disk(valid_dir)

        logger.info(f"Train set: {len(self.train_set):,} examples")
        logger.info(f"Validation set: {len(self.valid_set):,} examples")
        return

    def _initialize_dataloader(self) -> None:
        """Initialize PyTorch DataLoaders for training and validation.

        Creates DataLoader instances that handle batching and shuffling:
        - Training loader: Shuffled for better generalization
        - Validation loader: Not shuffled for consistent evaluation

        Returns:
            None
        """
        logger.info("Initializing Dataloader...")
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.valid_set, batch_size=self.batch_size, shuffle=False
        )

    def _initialize_model(self) -> None:
        """Initialize model, optimizer, scheduler, and loss function.

        Sets up all components needed for training:
        - Transformer model with specified architecture
        - Adam optimizer with configured hyperparameters
        - Linear warmup scheduler for learning rate
        - CrossEntropyLoss with label smoothing
        - Also logs model parameter counts for reference.

        Returns:
            None
        """
        # ==================== Model Initialization ====================
        logger.info("Initializing Model...")
        self.model: nn.Module = Transformer(
            source_vocab_size=self.vocab_size,
            target_vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )

        # Calculate and log parameter counts
        total_params = sum(param.numel() for param in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            f"Total number of parameters:\n"
            f"\tTrainable parameters: {trainable_params:,}\n"
            f"\tIncluding non-trainable): {total_params:,}"
        )

        # ==================== Optimizer Initialization ====================
        logger.info("Initializing Optimizer...")
        # Adam optimizer with configured hyperparameters
        self.optimizer: optim.Optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=self.lr,
            betas=self.betas,  # Exponential decay rates for moment estimates
            eps=float(self.eps),  # Term added to denominator for numerical stability
        )

        # ==================== Learning Rate Scheduler ====================
        logger.info("Initializing LR Scheduler...")
        # Linear warmup followed by linear decay
        self.lr_scheduler: optim.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,  # Steps to increase LR linearly
            num_training_steps=self.total_steps,  # Total steps for full schedule
        )

        # ==================== Loss Function ====================
        logger.info("Initializing Criterion...")
        # CrossEntropyLoss with label smoothing to prevent overconfidence
        # Label smoothing: instead of hard targets [0,0,1,0], uses softer [ε,ε,1-3ε,ε]
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def run(self) -> None:
        """Execute the complete training pipeline.

        Orchestrates the full training workflow:
        1. Setup output directories
        2. Load hyperparameters
        3. Load datasets
        4. Initialize dataloaders
        5. Initialize model components
        6. Create Trainer instance and begin training

        Returns:
            None
        """
        logger.info("Transformer Training Pipeline Starting...")

        # Setup phase
        self._saving_dir()
        self._hyperparameters()
        self._load_dataset()
        self._initialize_dataloader()
        self._initialize_model()

        # Create trainer instance with all components
        trainer = Trainer(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            total_epochs=self.total_epochs,
            total_steps = self.total_steps
            lr_scheduler=self.lr_scheduler,
            device=self.device,
            checkpoint_dir=self.checkpoint_dir,
            tensorboard_dir=self.tensorboard_dir,
            result_dir=self.result_dir,
        )
        # Execute training loop
        trainer.training()


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for training script.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - config (str): Path to YAML configuration file
            - model_type (str): Model architecture variant identifier
            - training_dataset (str): Dataset identifier from config
    """
    parser = argparse.ArgumentParser(
        description="Train Transformer model for neural machine translation."
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="transformer/config_files/config.yaml",
        help="Path to YAML configuration file with hyperparameters and settings.",
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default="base_model",
        help="Model architecture type (e.g., 'base_model', 'large_model').",
    )
    parser.add_argument(
        "--training_dataset",
        type=str,
        default="de_en",
        help="Dataset identifier key from config (e.g., 'de_en', 'fr_en').",
    )
    args = parser.parse_args()
    return args


def main():
    """Main entry point for training script."""
    start_time = time.time()

    # Parse arguments and load configuration
    args = get_args()

    # Load configuration from YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Initialize and execute training
    training = Training(
        config=config,
        model_type=args.model_type,
        training_dataset=args.training_dataset,
    )
    training.run()

    logger.info(f"Time taken: {get_elapsed_time(start_time)}")


if __name__ == "__main__":
    main()
