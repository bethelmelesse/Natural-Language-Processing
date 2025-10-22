import os
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import logging
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def log_result(
    epoch_loss: dict[str, list[float]],
    epoch_metrics: float,
    mode: str,
    lr: float,
) -> None:
    """Log the training results for the current epoch."""
    log_string = f"{mode} | "

    for key, value in epoch_loss.items():
        string = f"{key}: {value:.4f} | "
        log_string += string

    for key, value in epoch_metrics.items():
        string = f"{key}: {value:.4f} | "
        log_string += string

    if mode == "Train":
        log_string += f"lr: {lr}"

    logger.info(log_string)


def write_to_tensorboard(
    epoch: int,
    epoch_loss: dict[str, list[float]],
    writer: SummaryWriter,
    mode: str,
    epoch_metrics: float = None,
    lr: float = None,
) -> None:
    """Write the result to tensorboard."""
    for key, value in epoch_loss.items():
        writer.add_scalar(f"{key}/{mode}", value, epoch)

    if epoch_metrics is not None:
        for key, value in epoch_metrics.items():
            writer.add_scalar(f"{key}/{mode}", value, epoch)

    if mode == "Train":
        writer.add_scalar("Learning_Rate", lr, epoch)
    return


def write_to_csv(
    epoch: int,
    epoch_loss: dict[str, list[float]],
    epoch_metrics: float,
    result_dir: str,
    mode: str,
    lr: float,
) -> None:
    output_path = os.path.join(result_dir, f"{mode.lower()}_result.csv")
    row_data = {"Epoch": [epoch]}
    for key, value in epoch_loss.items():
        row_data[key] = [value]
    for key, value in epoch_metrics.items():
        row_data[key] = [value]
    if mode == "Train":
        row_data["LR"] = lr
    df_result = pd.DataFrame(row_data)
    if os.path.exists(output_path):
        # Append without header
        df_result.to_csv(output_path, mode="a", header=False, index=False)
    else:
        # Write with header (first time)
        df_result.to_csv(output_path, mode="w", header=True, index=False)
    return
