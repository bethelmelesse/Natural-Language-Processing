import torch


def get_input_data(batch_data) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract input and target data from batch."""
    source_ids = batch_data["input_ids"]
    source_mask = batch_data["attention_mask"]
    labels = batch_data["labels"]

    # Prepare decoder input and labels
    target_ids = labels[:, :-1]  # All token except the last
    # shift labels
    labels = labels[:, 1:]  # All token except the first

    # create target mask
    target_mask = (target_ids != -100).long()  # 1 where not padding

    return source_ids, target_ids, source_mask, target_mask, labels
