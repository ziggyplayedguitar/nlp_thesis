# Multilingual Troll Detection using DistilBERT

## Overview
This project implements a multilingual troll detection system using transformer-based models (DistilBERT/mBERT/XLM-R) to identify troll accounts based on their tweet content. The system is designed to work across different languages, with a particular focus on English training data and Czech test data.

## Project Structure
```
├── notebooks/                  # Jupyter notebooks for exploration and analysis
│   ├── 01_preprocess.ipynb     # Data preprocessing
│   ├── 02_train.ipynb          # Model training
│   ├── 03_benchmark.ipynb      # Model benchmark test
│   └── 04_visualize.ipynb      # Visualizing data and model output
├── src/  
│   ├── analysis/               # Model analysis utilities
│   │   ├── benchmark.py         # Benchmark to test model on same cases
│   ├── data_tools/             # Data processing utilities
│   │   ├── czech_data_tools.py # Novinky.cz data loading tools
│   │   ├── preprocessor.py     # Text preprocessing utilities
│   │   └── dataset.py          # Dataset creation utilities
│   ├── models/                 # Model definitions and training code
│   │   ├── troll_detector.py   # Transformer-based model
│   │   └── predictor.py        # Prediction and explanation utilities
├── output/                     # Ploomber generated output notebooks and other outputs
├── data/                       # Data storage (not in repo)
│   ├── MediaSource/            # Novinky.cz comments, unlabeled
│   ├── russian_troll_tweets/   # Russian troll tweet dataset
│   ├── sentiment140/           # Twitter regular user dataset
│   ├── information_operations  # Information operations troll dataset
│   ├── non_troll_politics/     # Political non-troll twitter accounts
│   ├── celebrity_tweets/       # Verified non-troll accounts
│   └── processed/              # Processed data files
├── checkpoints/                # Saved model checkpoints
├── pipeline.yaml               # Python task definitions for Ploomber
├── tasks.py                    # Ploomber pipeline configuration
```

## Features
- **Czech Media Comment Analysis**: Focuses on detecting trolls in Czech online news comment sections.
- **Ploomber Pipeline**: Structured workflow for reproducible machine learning execution.
- **Data Preprocessing**: Cleans and formats tweets, removes unnecessary elements like URLs, and organizes data for training.
- **Custom Dataset**: Structures tweets and comments at an account level rather than tweet-by-tweet classification.
- **Multi-Tweet Attention Mechanism**: Assigns different weights to tweets when making an account-level decision.
- **Transformer-based Model**: Utilizes pre-trained DistilBERT or XLM-R for feature extraction.
- **Multi-stage Processing**: Separate preprocessing, training, evaluation and prediction steps.
- **Configurable Training Pipeline**: Implements gradient clipping, learning rate scheduling, and mixed precision training.
- **Prediction & Explainability**: Provides an interface for making predictions and generating explanations for model decisions.

## Data Sources
The project uses multiple data sources:
1. Russian Troll Tweets (labeled troll accounts)
2. Sentiment140 Dataset (regular Twitter users)
3. Celebrity Tweets (verified non-troll accounts)
4. Information Operations (additional troll dataset)
5. Political Non-troll Tweets (tweets from legitimate accounts)
6. Czech Novinky.cz discussion comments (Main dataset of comments from Czech newssite novinky.cz)

## Installation
### Prerequisites
- Python 3.8+
- ploomber
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- NumPy
- Pandas
- tqdm
- wandb (optional for experiment tracking)
- matplotlib
- seaborn
- jupyter
- pathlib
- papermill

```bash
pip install -r requirements.txt
```

## Usage

### Running the Pipeline
To execute the full pipeline:
```bash
ploomber build
```

To run a specific task:
```bash
ploomber task <task_name>
```

## Configuration
Key configuration parameters in `train.py`:
```python
config = {
    'model_name': 'distilbert-base-multilingual-cased',
    'max_length': 128,
    'batch_size': 64,
    'learning_rate': 2e-5,
    'weight_decay': 0.03,
    'num_epochs': 3,
    'dropout_rate': 0.2,
    'warmup_steps': 50,
    'max_grad_norm': 1.0,
    'comments_per_user': 5,
    'early_stopping_patience': 3,
    'use_wandb': False,
    'random_state': 17,
    'label_smoothing': 0.1       
}
```

## Model Architecture
- **Transformer Encoder**: Extracts contextual embeddings from each tweet.
- **Tweet-Level Attention**: Determines the importance of each tweet in an account.
- **Account-Level Classifier**: Aggregates tweet embeddings and predicts troll likelihood.

## Evaluation
The model is evaluated on:
- Account-level classification accuracy
- ROC-AUC score
- Precision, Recall, and F1-score

## Future Improvements
- Fine-tuning with more diverse datasets
- Adding Czech test set evaluation

## Acknowledgments
- Hugging Face Transformers
- PyTorch community
