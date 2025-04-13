import streamlit as st
import pandas as pd
import json
from pathlib import Path

from src.models.predictor import TrollPredictor 

# --- CONFIG ---
COMMENTS_PARQUET = "./data/processed/czech_media_comments.parquet"
ANOMALY_CSV = "./user_anomaly_scores.csv"
ANNOTATIONS_OUTPUT = "./annotations/user_labels.csv"
Path("./annotations").mkdir(exist_ok=True)

# Predictor from Checkpoint
predictor = TrollPredictor(
    model_path= "./checkpoints/best_model.pt",
    comments_per_user=50,
    max_length=96
)

# --- Load Data ---
@st.cache_data
def load_data():
    df_comments = pd.read_parquet(COMMENTS_PARQUET)
    df_anomaly = pd.read_csv(ANOMALY_CSV)
    return df_comments, df_anomaly

df_comments, df_anomaly = load_data()

# --- Author Selector ---
# Load existing annotations
if Path(ANNOTATIONS_OUTPUT).exists():
    df_labels = pd.read_csv(ANNOTATIONS_OUTPUT)
    labeled_authors = set(df_labels["author"].tolist())
else:
    df_labels = pd.DataFrame()
    labeled_authors = set()

# Authors not yet labeled
authors = [a for a in df_anomaly["author"].tolist() if a not in labeled_authors]
selected_author = st.selectbox("Select an Author", authors)

# --- Display Info ---
author_info = df_anomaly[df_anomaly["author"] == selected_author].iloc[0]
st.markdown(f"### Author: `{selected_author}`")

# Get author's comments for prediction
author_comments = df_comments[df_comments["author"] == selected_author]["text"].tolist()

# Get troll prediction
pred_result = predictor.predict(author_comments)
prediction = pred_result["prediction"]
confidence = pred_result["confidence"]

# Display all scores with color coding
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Troll Classification",
        value=prediction,
        delta=f"{confidence:.3f} confidence"
    )

with col2:
    st.metric(
        label="Isolation Forest Score",
        value=f"{author_info['anomaly_score_iforest']:.3f}",
        delta="higher = more anomalous"
    )

with col3:
    st.metric(
        label="LOF Score",
        value=f"{author_info['anomaly_score_lof']:.3f}",
        delta="higher = more anomalous"
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
                
                # Show truncated comment if too long
                if len(comment_data['text']) > 200:
                    st.markdown(f"{comment_data['text'][:200]}...")
                else:
                    st.markdown(comment_data['text'])
                st.markdown("---")

# --- Labeling ---
label = st.radio("Label this author as:", ["Uncertain", "Not Troll", "Troll"])

# --- Save Label ---
if st.button("Save Label"):
    label_map = {"Uncertain": -1, "Not Troll": 0, "Troll": 1}
    label_data = {
        "author": selected_author,
        "iforest_score": author_info["anomaly_score_iforest"],
        "lof_score": author_info["anomaly_score_lof"],
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
