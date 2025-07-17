# Multilingual Troll Detection using Transformer Models

## Overview
This project implements a multilingual troll detection system using transformer-based models to identify troll accounts based on their comment content. The system is designed to work across different languages, with a particular focus on English and Russian training data and Czech test data. The project specifically analyzes comments from Czech news discussion sections to detect potential troll behavior.

## Project Structure
```
├── notebooks/                  # Jupyter notebooks for exploration and analysis
│   ├── 01_preprocess.ipynb     # Data preprocessing
│   ├── 02_train.ipynb          # Model training
│   ├── 03_fine_tune.ipynb      # Model fine-tuning on Czech dataset
│   ├── 04_predict.ipynb        # Using trained models to predict Czech news comments
│   ├── 05_stylometry.ipynb     # Baseline benchmark against classical ML models
│   ├── 07_adapter.ipynb        # Failed experiment of training a Czech language adapter
│   └── 08_analyze.py           # Analysis of attention mechanism
├── src/  
│   ├── analysis/               # Model analysis utilities
│   │   └── user_analysis.py    # User behavior analysis tools
│   ├── data_tools/             # Data processing utilities
│   │   ├── czech_data_tools.py # Novinky.cz data loading tools
│   │   ├── preprocessor.py     # Text preprocessing utilities
│   │   ├── dataset.py          # Dataset creation utilities
│   │   └── examine_parquet.py  # Parquet file inspection utilities
│   ├── models/                 # Model definitions and training code
│   │   ├── bert_model.py       # Transformer model architecture
│   │   ├── predictor.py        # Prediction and explanation utilities
│   │   ├── trainer.py          # Model training utilities
│   │   └── train_adapter.py    # Language adapter training utilities
├── data/                       # Data storage (not in repo)
│   ├── MediaSource/            # Novinky.cz comments, unlabeled
│   ├── russian_troll_tweets/   # Russian troll tweet dataset
│   ├── sentiment140/           # Twitter regular user dataset
│   ├── information_operations  # Information operations troll dataset
│   ├── non_troll_politics/     # Political non-troll twitter accounts
│   ├── celebrity_tweets/       # Verified non-troll accounts
│   └── processed/              # Processed data files
├── output/                     # Generated prediction files
│   └── *predictions*.csv       # CSV files containing model predictions
└── checkpoints/                # Saved model checkpoints
```

The `/output` directory contains pre-generated CSV files with model predictions. These files can be loaded directly for analysis without needing to run the prediction process again. This is especially useful for analyzing large datasets or when you want to compare results across different experiments.

## Features
- **Czech Media Comment Analysis**: Advanced detection of trolls in Czech online news comment sections
- **Multi-Comment Processing**: Analyzes multiple comments per user for more accurate account-level classification
- **Transformer Architecture**: Uses state-of-the-art transformer models for text processing
- **Multilingual Capability**: Supports English, Russian, and Czech language content
- **User Behavior Analysis**: Tools for analyzing user commenting patterns and behavior
- **Flexible Model Training**: Supports various transformer architectures with customizable training parameters
- **Explainable Predictions**: Provides insight into model decisions through attention visualization

## Data Processing Pipeline
1. **Data Collection**: Gathering comments from Novinky.cz and other sources
2. **Preprocessing**: Text cleaning and standardization using `preprocessor.py`
3. **Dataset Creation**: Building structured datasets with `dataset.py`
4. **Model Training**: Training using `trainer.py` with configurable parameters
5. **Prediction**: Making predictions using `predictor.py`
6. **Analysis**: Analyzing results with tools in the `analysis` directory

## Model Architecture
The project implements a hierarchical transformer architecture:
- **Base Transformer**: Processes individual comments using pre-trained models
- **Comment-Level Attention**: Weighs the importance of different parts within comments
- **User-Level Aggregation**: Combines multiple comments to make user-level predictions

## Installation
### Prerequisites
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- pandas
- numpy
- scikit-learn
- tqdm

## Configuration
Key configuration parameters in `02_train.ipynb`:
```python

# Updated training configuration
config = {
    # Model configuration
    'model_name': 'distilbert-base-multilingual-cased',
    'adapter_path': None,
    # Data parameters
    'max_length': 96,
    'batch_size': 16,
    # Training hyperparameters
    'learning_rate': 1e-5,
    'weight_decay': 0.01,
    'num_epochs': 6,
    'dropout_rate': 0.2,
    'warmup_steps': 50,
    'max_grad_norm': 1.0,
    'comments_per_user': 10,
    # Training control
    'early_stopping_patience': 3,
    'random_state': 17,
}
```

## Acknowledgments
- Hugging Face Transformers
- PyTorch community

## Preprocessing Script
Use `scripts/preprocess.py` to prepare the dataset outside of the notebooks.
The script loads raw files, creates author-based splits and saves each split as
parquet files.

```bash
python scripts/preprocess.py \
    --input-dir /path/to/raw/data \
    --output-dir /path/to/output \
    --random-seed 42
```

This command generates `train.parquet`, `val.parquet` and `test.parquet` in the
output directory. Additional arguments like `--train-size`, `--val-size`, and
`--test-size` control the split ratios, while `--max-tweets-per-source` and
`--max-tweets-per-author` limit dataset size. Run the script with `--help` for a
full list of options.
