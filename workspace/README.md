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
    'max_length': 64,
    'batch_size': 64,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'num_epochs': 3,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'comments_per_user': 5,
    'use_wandb': False,
    'random_state': 42  # Default if config not found
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

## File Structure

File: preprocess_data.py

Class: TweetPreprocessor

Purpose: Cleans tweet texts by removing URLs, picture links, and extra spaces.

Method:

preprocess_tweet(text: str): Cleans an individual tweet's text, removes links, and extra whitespace.

Functions:

load_and_clean_data(data_dir: str): Loads tweet datasets (Russian troll tweets, Sentiment140, and celebrity tweets), merges them, labels troll accounts, and filters accounts with too few tweets.

create_data_splits(df: pd.DataFrame, train_ratio: float, val_ratio: float): Splits data into training, validation, and test datasets, ensuring each account is in only one set.

collate_batch(batch: List[Dict[str, torch.Tensor]]): Prepares batches of tweet data for model training by stacking tweet data and labels.

Class:

TrollTweetDataset

A custom dataset class that groups tweets by account and formats them for training.

Methods:

__len__: Counts the number of accounts in the dataset.

__getitem__(idx: int): Retrieves processed tweet data and labels for each account, ensuring a fixed number of tweets per account.

File: model.py

Classes:

TrollDetector

Implements a neural network model using DistilBERT to detect troll accounts based on tweet text.

Methods:

forward(input_ids, attention_mask, tweets_per_account): Processes tweet inputs, calculates importance weights (attention), aggregates tweet representations, and predicts whether an account is a troll or not.

predict(input_ids, attention_mask, tweets_per_account): Predicts if an account is a troll, returning predictions and confidence scores.

TrollDetectorWithPooling

Similar to TrollDetector, but uses simpler pooling techniques (mean or max) instead of attention.

Methods:

forward(input_ids, attention_mask, tweets_per_account): Processes tweet embeddings with pooling.

predict(input_ids, attention_mask, tweets_per_account): Generates predictions and probabilities.

File: train.py

Class:

TrollDetectorTrainer

Handles the training and evaluation of the troll detection model.

Key methods:

train_epoch(): Trains the model for one epoch, handles gradient scaling, and updates model weights.

evaluate(dataloader): Evaluates model performance on validation or test sets.

calculate_metrics(preds, labels, probs=None): Calculates accuracy, precision, recall, F1-score, and ROC-AUC.

save_checkpoint(epoch, metrics, is_best=False): Saves the trained model checkpoints.

train(): Runs the full training cycle for multiple epochs and evaluates the model performance.

Main Function:

main(): Sets up data loaders, initializes the model and trainer with configurations, and starts training.

File: predict.py

Class:

TrollPredictor

Loads the trained model to predict if accounts are trolls based on their tweets.

Methods:

preprocess_tweets(tweets): Preprocesses a list of tweets.

prepare_input(tweets): Prepares and tokenizes tweets for prediction.

predict(tweets): Predicts whether an account is a troll and provides confidence scores and attention details if available.

explain_prediction(tweets): Generates explanations highlighting influential tokens using occlusion sensitivity analysis.

explain_with_correlation(tweets): Provides explanations based on token correlation with model predictions.

Main function (main()):

Handles command-line arguments, prepares input data, performs predictions, and optionally generates explanations.

Each component works together to preprocess data, train a DistilBERT-based classifier to detect troll accounts from tweets, and provide insights into model predictions through explainability methods.

## Acknowledgments
- Hugging Face Transformers
- PyTorch community
