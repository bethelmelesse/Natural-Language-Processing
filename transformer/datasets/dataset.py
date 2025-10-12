from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

import logging
import os
import yaml
import argparse
import sys
import time


current_dir = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
greatgrandparent_dir = os.path.dirname(grandparent_dir)

sys.path.append(greatgrandparent_dir)
from utils.utils import get_elapsed_time


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetPreparation:
    """Handles dataset loading and tokenization for Transformer models.

    This class manages the complete pipeline for preparing translation datasets:
    1. Loading datasets from HuggingFace or local cache
    2. Initializing pretrained tokenizers
    3. Tokenizing and caching processed datasets"""

    def __init__(self, config, training_dataset) -> None:
        """
        Initialize the DatasetPreparation instance.

        Args:
            config (dict): Configuration dictionary with dataset and training parameters
            training_dataset (str): Key identifying which dataset config to use
        Returns:
            None
        """
        self.config = config
        self.training_dataset = training_dataset
        self._load_config()

    def _load_config(self) -> None:
        """Load and parse configuration parameters from the config dictionary.

        Returns:
            None
        """
        # Directory paths for data storage
        data: dict = self.config["data"]
        self.data_dir: str = data["data_dir"]
        self.batch_size: int = data["batch_size"]
        self.max_len: int = data["max_len"]
        # Dataset size control: subset allows using only a fraction of data (e.g., 0.1 = 10%)
        self.subset: float | None = data["subset"]
        tokenized_data_dir = data["tokenized_data_dir"]

        # Extract dataset-specific configuration
        dataset: dict = self.config[self.training_dataset]

        # HuggingFace dataset identifiers
        self.hf_dataset_path: str = dataset["hf_dataset_path"]  # e.g., "wmt14"
        self.dataset_name: str = dataset["dataset_name"]  # e.g., "de-en"
        self.tokenizer_path: str = dataset["tokenizer_path"]
        # Language pair configuration
        self.source_language: str = dataset["source_language"]
        self.target_language: str = dataset["target_language"]

        # Tokenized data directory: Append subset percentage to directory name if using subset
        self.tokenized_data_dir: str = os.path.join(
            tokenized_data_dir, self.dataset_name
        )
        if self.subset:
            self.tokenized_data_dir += "_" + str(self.subset * 100)

        return

    def _load_dataset(self) -> None:
        """Load dataset from local cache or download from HuggingFace.

        - Implements intelligent caching: checks for existing local dataset first, downloads only if necessary.
        - Splits dataset into train/validation/test sets.

        Returns:
            None
        """
        dataset_dir = os.path.join(self.data_dir, self.dataset_name)
        logger.info(f"Checking for cached dataset in {dataset_dir}...")

        # Check if the dataset has already been saved locally
        if os.path.exists(dataset_dir):
            logger.info("Local dataset found. Loading from disk...")
            dataset = load_from_disk(dataset_dir)
        else:
            logger.info("No local dataset found. Downloading and saving...")
            # Create data directory if it doesn't exist
            os.makedirs(self.data_dir, exist_ok=True)
            # Download dataset from HuggingFace
            dataset = load_dataset(
                path=self.hf_dataset_path,
                name=self.dataset_name,
                cache_dir=self.data_dir,
            )
            # Save to disk for future use
            dataset.save_to_disk(dataset_dir)
            logger.info(f"Dataset saved to {dataset_dir}")

        # Store dataset splits with standardized naming
        self.dataset_splits = {
            "train": dataset["train"],
            "valid": dataset["validation"],
            "test": dataset["test"],
        }
        return

    def _initialize_tokenizer(self) -> AutoTokenizer:
        """Initialize the tokenizer from a pretrained model.

        Loads a pretrained tokenizer (e.g., from HuggingFace) that will be used to convert text into token IDs for the transformer model.

        Returns:
            AutoTokenizer: The initialized tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return self.tokenizer

    def tokenize_function(self, batch: dict) -> dict:
        """Tokenize a batch of translation pairs.

        - This function is designed to be used with the HuggingFace datasets.map() function.
        - It extracts source and target texts from the translation pairs and tokenizes them with appropriate truncation.

        Args:
            batch (dict): A batch of examples with 'translation' field containing source and target language pairs

        Returns:
            dict: Tokenized batch containing input_ids, attention_mask, and labels for both source and target sequences
        """
        # Extract source and target language texts from batch
        source_texts = [
            example[self.source_language] for example in batch["translation"]
        ]
        target_texts = [
            example[self.target_language] for example in batch["translation"]
        ]

        # Tokenize both source and target texts
        tokenize_fn = self.tokenizer(
            source_texts,
            text_target=target_texts,
            truncation=True,
            max_length=self.max_len,
        )
        return tokenize_fn

    def _tokenize_dataset(self) -> None:
        """Tokenize all dataset splits and save to disk.

        - Processes each split (train/valid/test) through tokenization if not already cached.
        - Supports subset selection for faster experimentation.
        - Uses batched processing for efficiency.

        Returns:
            None
        """
        for split_name, split_set in self.dataset_splits.items():
            # Determine output directory path
            tokenized_split_dir = os.path.join(self.tokenized_data_dir, split_name)

            # Check if tokenized dataset already exists
            if os.path.exists(tokenized_split_dir):
                logger.info(f"Tokenized dataset found for {split_name}. Skipping...")
            else:
                logger.info(f"No local dataset found for {split_name}. Tokenizing...")

                #  Apply subset selection if specified
                if self.subset:
                    total_samples = len(split_set)
                    num_examples = int(total_samples * self.subset)

                    # Select first N examples based on subset fraction
                    split_set = split_set.select(range(num_examples))

                    logger.info(
                        f"Using {num_examples:,} examples ({self.subset * 100}%) from {split_name} split with {total_samples:,}."
                    )

                # Apply tokenization function to entire dataset in batches
                tokenized_dataset = split_set.map(
                    self.tokenize_function, batched=True, batch_size=self.batch_size
                )

                # Save tokenized dataset to disk
                os.makedirs(tokenized_split_dir, exist_ok=True)
                tokenized_dataset.save_to_disk(tokenized_split_dir)
                logger.info(f"Tokenized {split_name} saved to {tokenized_split_dir}")
        return

    def __call__(self) -> None:
        """Execute the complete dataset preparation pipeline.

        This method orchestrates the full workflow:
        1. Load or download the dataset
        2. Initialize the tokenizer
        3. Tokenize and cache all dataset splits

        Returns:
            None
        """
        self._load_dataset()
        self._initialize_tokenizer()
        self._tokenize_dataset()
        return


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the dataset preparation script.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - config (str): Path to YAML configuration file
            - training_dataset (str): Dataset identifier from config
    """
    parser = argparse.ArgumentParser(
        description="Prepare and tokenize datasets for Transformer model training."
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="transformer/config_files/config.yaml",
        help="Path to YAML configuration file containing hyperparameters and dataset settings.",
    )
    parser.add_argument(
        "--training_dataset",
        type=str,
        default="fr_en",
        help="Dataset identifier key from the config file (e.g., 'de_en', 'fr_en').",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    """Main execution block for dataset preparation."""
    start_time = time.time()
    # Parse command-line arguments
    args = get_args()

    # Load configuration from YAML file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Initialize dataset preparation handler
    prepare_data = DatasetPreparation(
        config=config, training_dataset=args.training_dataset
    )

    # Execute the preparation pipeline
    prepare_data()
    logger.info(f"Time taken: {get_elapsed_time(start_time)}")
    return


if __name__ == "__main__":
    main()
