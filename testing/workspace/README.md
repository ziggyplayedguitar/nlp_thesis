# Multilingual Troll Detection using DistilBERT

## Overview
This project implements a multilingual troll detection system using transformer-based models (DistilBERT/mBERT/XLM-R) to identify troll accounts based on their tweet content. The system is designed to work across different languages, with a particular focus on English training data and Czech test data.

## Project Structure
```
├── preprocess_data.py   # Data preprocessing and dataset creation
├── train.py             # Training script for the model
├── model.py             # Model definition using DistilBERT/mBERT/XLM-R
├── predict.py           # Prediction and explanation script
├── data/                # Directory containing dataset files
├── checkpoints/         # Directory for saving trained models
└── README.md            # Project documentation
```

## Features
- **Multi-lingual Support**: Trained on multilingual but mostly English data.
- **Data Preprocessing**: Cleans and formats tweets, removes unnecessary elements like URLs, and organizes data for training.
- **Custom Dataset**: Structures tweets at an account level rather than tweet-by-tweet classification.
- **Multi-Tweet Attention Mechanism**: Assigns different weights to tweets when making an account-level decision.
- **Transformer-based Model**: Utilizes pre-trained DistilBERT, mBERT, or XLM-R for feature extraction.
- **Configurable Training Pipeline**: Implements gradient clipping, learning rate scheduling, and mixed precision training.
- **Prediction & Explainability**: Provides an interface for making predictions and generating explanations for model decisions.

## Data Sources
The project uses multiple data sources:
1. Russian Troll Tweets (labeled troll accounts)
2. Sentiment140 Dataset (regular Twitter users)
3. Celebrity Tweets (verified non-troll accounts)

## Installation
### Prerequisites
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- NumPy
- Pandas
- tqdm
- wandb (optional for experiment tracking)

## Usage

### Training the Model
To train the model, run:
```bash
python train.py
```
This will:
- Load and preprocess data
- Train DistilBERT/mBERT/XLM-R with attention layers
- Save the best model checkpoints

### Making Predictions
To predict whether an account is a troll based on tweets:
```bash
python predict.py --model_path checkpoints/best_model.pt --input_file "tweets_file.json"
```
or

```bash
python predict.py --model_path checkpoints/best_model.pt --tweets "tweet1" "tweet2" ...
```

### Explainability
To get explanations for predictions:
```bash
python predict.py --model_path checkpoints/best_model.pt --input_file "tweets_file.json" --explain
```
This runs occlusion-based analysis to highlight important tokens in the prediction.

## Configuration
Key configuration parameters in `train.py`:
```python
config = {
    'model_name': "bert-base-multilingual-cased",
    'max_length': 128,
    'tweets_per_account': 10,
    'batch_size': 8,
    'learning_rate': 2e-5,
    'num_epochs': 10,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'use_wandb': False,
    'data_dir': "/workspace/data",
    'min_tweets_per_account': 10,
    'train_ratio': 0.8,
    'val_ratio': 0.1
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
