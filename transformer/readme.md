# Transformers

## 🚀 Step 1: Data Preparation

The data preparation stage handles downloading, tokenization, and saving the data in a format ready for training.

- Python Script: `transformer/datasets/dataset.py`

- Config File: `transformer/config_files/config.yaml`

- Supported Training Datasets
    1. de_en – English ↔ German
    2. fr_en – English ↔ French

### Running Command
```bash
python <path_to_python_script> \
    --config <path_to_config_file> \
    --training_dataset <de_en|fr_en>
```

### Example Usage
```bash
python transformer/datasets/dataset.py \
    --config transformer/config_files/config.yaml \
    --training_dataset de_en
```

### Config File content
```YAML
data:
    max_len: 512                # Maximum token length per sentence
    batch_size: 10000           # Number of examples per batch
    subset: 0.025               # Fraction of dataset to use for testing/debug
    data_dir: "data"            # Raw data storage
    tokenized_data_dir: "data/tokenized"  # Where tokenized data will be saved

de_en:
    source_language: "en"       # Source language
    target_language: "de"       # Target language
    hf_dataset_path: "wmt14"    # Hugging Face dataset path
    dataset_name: "de-en"       # Dataset identifier
    tokenizer_path: "Helsinki-NLP/opus-mt-en-de"  # Pretrained tokenizer

fr_en:
    source_language: "en"
    target_language: "fr"
    hf_dataset_path: "wmt14"
    dataset_name: "fr-en"
    tokenizer_path: "Helsinki-NLP/opus-mt-en-fr"
    
```

#### Directory Structure
```arduino
    transformer/
    │
    ├─ datasets/
    │   └─ dataset.py
    ├─ config_files/
    │   └─ config.yaml
    └─ data/
        └─ tokenized/
```