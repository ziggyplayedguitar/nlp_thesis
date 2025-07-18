import streamlit as st
import pandas as pd
import json
from pathlib import Path
import torch

from src.models.predictor import TrollPredictor 

# --- CONFIG ---
COMMENTS_PARQUET = "./data/processed/czech_media_comments.parquet"
ANNOTATIONS_OUTPUT = "./annotations/user_labels.csv"
Path("./annotations").mkdir(exist_ok=True)

# Initialize predictor with caching
@st.cache_resource
def get_predictor():
    try:
        return TrollPredictor(
            # model_path="./checkpoints/best_model_english_medium",
            model_path="./checkpoints/best_model_ru_only_finetuned_enhanced_attention.pt",
            # model_name="distilbert-base-multilingual-cased",
            comments_per_user=20,
            max_length=96,
            use_adapter=False,
            threshold=0.4,
        )
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

# Initialize predictor
predictor = get_predictor()
if predictor is None:
    st.error("Failed to initialize the model. Please check the model path and configuration.")
    st.stop()

# --- Load Data ---
@st.cache_data
def load_data():
    try:
        df_comments = pd.read_parquet(COMMENTS_PARQUET)
        return df_comments
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df_comments = load_data()
if df_comments is None:
    st.error("Failed to load data. Please check the data paths.")
    st.stop()

# --- Author Selector ---
# Load existing annotations
if Path(ANNOTATIONS_OUTPUT).exists():
    df_labels = pd.read_csv(ANNOTATIONS_OUTPUT)
    labeled_authors = set(df_labels["author"].tolist())
else:
    df_labels = pd.DataFrame()
    labeled_authors = set()

# Authors not yet labeled
# authors = [a for a in df_anomaly["author"].tolist() if a not in labeled_authors]
authors = df_comments["author"].tolist()
selected_author = st.selectbox("Select an Author", authors)

# --- Display Info ---
author_info = df_comments[df_comments["author"] == selected_author].iloc[0]
st.markdown(f"### Author: `{selected_author}`")

# Get author's comments for prediction
author_comments = df_comments[df_comments["author"] == selected_author]["text"].tolist()

# Get troll prediction
pred_result = predictor.predict(author_comments)
prediction = pred_result["prediction"]  # This is still 'troll' or 'not_troll' based on threshold
trolliness_score = pred_result["trolliness_score"]  # New continuous score
binary_confidence = pred_result["binary_confidence"]  # New confidence metric

# Display scores with color coding
col1, col2 = st.columns(2)  # Changed to 2 columns

with col1:
    st.metric(
        label="Trolliness Score",
        value=f"{trolliness_score:.3f}",
        delta=f"Threshold: {predictor.threshold}"
    )

with col2:
    st.metric(
        label="Binary Classification",
        value=prediction,
        delta=f"{binary_confidence:.3f} confidence"
    )

# --- Comments Preview ---
st.markdown("#### Example Comments")

# Get comments with their article titles
author_comments_with_articles = df_comments[df_comments["author"] == selected_author][["text", "article_title", "article_id"]].to_dict('records')
author_comments_raw = [comment["text"] for comment in author_comments_with_articles]

# Run prediction with attention weights - no need to pad, predictor handles this
pred = predictor.predict(author_comments_raw)

# If attention weights available, sort comments
if "attention_weights" in pred:
    attention = pred["attention_weights"]
    # Only zip with the original comments and articles, not padded ones
    ranked = sorted(zip(author_comments_with_articles[:len(attention)], attention), 
                   key=lambda x: x[1], reverse=True)
else:
    ranked = [(c, None) for c in author_comments_with_articles]

# Create a grid layout (3 columns)
COLS_PER_ROW = 3
num_comments = len(ranked)
num_rows = (num_comments + COLS_PER_ROW - 1) // COLS_PER_ROW  # Ceiling division

st.markdown("#### Top Comments (Ranked by Attention)")

# Create rows of comments
for row in range(num_rows):
    cols = st.columns(COLS_PER_ROW)
    
    # Fill each column in the row
    for col in range(COLS_PER_ROW):
        idx = row * COLS_PER_ROW + col
        
        if idx < num_comments:
            comment_data, attn = ranked[idx]
            with cols[col]:
                st.container()
                # Show comment number and attention weight if available
                if attn is not None:
                    st.markdown(f"**Comment {idx+1}** (Attention: **{attn:.4f}**)")
                else:
                    st.markdown(f"**Comment {idx+1}**")
                
                # Show article title and ID
                st.markdown(f"*Article Title: {comment_data['article_title']}*")
                
                # Show full comment
                st.markdown(comment_data['text'])
                st.markdown("---")

# --- Labeling ---
label = st.radio("Label this author as:", ["Uncertain", "Not Troll", "Troll"])

# --- Save Label ---
if st.button("Save Label"):
    label_map = {"Uncertain": -1, "Not Troll": 0, "Troll": 1}
    label_data = {
        "author": selected_author,
        "trolliness_score": trolliness_score,  # Add the continuous score
        "binary_confidence": binary_confidence,  # Add the binary confidence
        "label": label_map[label]
    }

    # Load existing
    if Path(ANNOTATIONS_OUTPUT).exists():
        df_labels = pd.read_csv(ANNOTATIONS_OUTPUT)
        df_labels = df_labels[df_labels["author"] != selected_author]  # Remove if exists
    else:
        df_labels = pd.DataFrame()

    df_labels = pd.concat([df_labels, pd.DataFrame([label_data])], ignore_index=True)
    df_labels.to_csv(ANNOTATIONS_OUTPUT, index=False)
    st.success(f"Labeled {selected_author} as {label}.")
    st.rerun()
