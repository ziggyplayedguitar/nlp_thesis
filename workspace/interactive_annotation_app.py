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
    comments_per_user=30,
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
st.write(f"**Isolation Forest Score:** {author_info['anomaly_score_iforest']:.3f}")
st.write(f"**LOF Score:** {author_info['anomaly_score_lof']:.3f}")

# --- Comments Preview ---
st.markdown("#### Example Comments")
author_comments_raw = df_comments[df_comments["author"] == selected_author]["text"].tolist()

# Get top N comments based on attention (pad if necessary)
comments = (author_comments_raw * ((predictor.comments_per_user // len(author_comments_raw)) + 1))[:predictor.comments_per_user]

# Run prediction with attention weights
pred = predictor.predict(comments)

# If attention weights available, sort comments
if "attention_weights" in pred:
    attention = pred["attention_weights"]
    ranked = sorted(zip(comments, attention), key=lambda x: x[1], reverse=True)
else:
    ranked = [(c, None) for c in comments]

st.markdown("#### Top Comments (Ranked by Attention)")
for i, (comment, attn) in enumerate(ranked):
    st.markdown(f"**{i+1}.** {comment}")
    if attn is not None:
        st.markdown(f"*Attention weight: {attn:.3f}*")
    st.markdown("---")

if attn is not None:
    st.progress(min(attn, 1.0))  # clamp in case of overflows

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
