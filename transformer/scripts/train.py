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

current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
greatgrandparent_dir = os.path.dirname(grandparent_dir)

sys.path.append(greatgrandparent_dir)

from utils.utils import get_elapsed_time
from transformer.scripts.trainer import Trainer
from transformer.model.transformer import Transformer


logger = logging.getLogger(__name__)


class Training:
    def __init__(self, config: dict, model_type: str):
        self.config = config
        self.model_type = model_type

    def _saving_dir(self):
        # Time and date
        date_time_str = datetime.now().strftime("%m-%d_%H-%M-%S")
        output_dir: dict = self.config["output_dir"]
        base_dir: str = os.path.join(
            output_dir["base_dir"], "transformer", self.model_type, date_time_str
        )
        self.checkpoint_dir = os.path.join(base_dir, output_dir["checkpoint_dir"])
        self.tensorboard_dir = os.path.join(base_dir, output_dir["tensorboard_dir"])
        self.result_dir = os.path.join(base_dir, output_dir["result_dir"])

        for directories in [self.checkpoint_dir, self.tensorboard_dir, self.result_dir]:
            os.makedirs(directories, exist_ok=True)

    def _hyperparameters(self):
        # Model hyperparameter
        model_hparams: dict = self.config[self.model_type]
        self.source_vocab_size: int = model_hparams["source_vocab_size"]
        self.target_vocab_size: int = model_hparams["target_vocab_size"]
        self.max_seq_len: int = model_hparams["max_seq_len"]
        self.d_model: int = model_hparams["d_model"]
        self.d_ff: int = model_hparams["d_ff"]
        self.num_layers: int = model_hparams["num_layers"]
        self.num_heads: int = model_hparams["num_heads"]
        self.dropout: float = model_hparams["dropout"]

        # Optimizer
        optimizer_hparams: dict = self.config[self.optimizer]
        self.lr: float = optimizer_hparams["lr"]
        self.betas: tuple[float, float] = optimizer_hparams["betas"]
        self.eps: float = optimizer_hparams["eps"]

        # Scheduler
        self.warmup_steps: float = self.config["scheduler"]["warmup_steps"]

        # loss
        self.label_smoothing: float = self.config["loss"]["label_smoothing"]

        # training
        training_hparams: dict = self.config["training"]
        cuda: int = training_hparams["cuda"]
        self.batch_size: int = training_hparams["batch_size"]
        self.total_epochs: int = training_hparams["total_epochs"]
        self.total_steps: int = training_hparams["total_steps"]

        # Device setup
        self.device = f"{cuda}" if torch.cuda.is_available() else "cpu"

    def _load_dataset(self):
        self.train_dataset, self.val_dataset = None, None
        return self.train_dataset, self.val_dataset

    def _initialize_dataloader(self):
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def _initialize_model(self):
        # Model
        self.model = Transformer(
            source_vocab_size=self.source_vocab_size,
            target_vocab_size=self.target_vocab_size,
            max_seq_len=self.max_seq_len,
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )

        # Optimizer
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
        )

        # Scheduler
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

        # Criterion dict
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

    def run(self):
        self._saving_dir()
        self._hyperparameters()
        self._load_dataset()
        self._initialize_dataloader()
        self._initialize_model()

        trainer = Trainer(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            total_epochs=self.total_epochs,
            lr_scheduler=self.lr_scheduler,
            device=self.device,
            checkpoint_dir=self.checkpoint_dir,
            tensorboard_dir=self.tensorboard_dir,
            result_dir=self.result_dir,
        )
        trainer.training()


def get_args():
    parser = argparse.ArgumentParser(description="Transformer model training.")

    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="transformer/config_files/config.yaml",
        help="Configuration File containing hyperparameters.",
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default="base_model",
        help="Model Type",
    )
    args = parser.parse_args()
    return args


def main():
    start_time = time.time()

    args = get_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    Training(config=config, model_type=args.model_type).run()

    logger.info(f"Time taken: {get_elapsed_time(start_time)}")


if __name__ == "__main__":
    main()
