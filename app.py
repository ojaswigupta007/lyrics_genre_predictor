import streamlit as st
import pickle
import re
from scipy.sparse import hstack

# ---------------------------------------
# Page setup
# ---------------------------------------
st.set_page_config(page_title="Song Genre Predictor", layout="centered")

st.title("🎵 Song Genre Prediction")
st.write("Paste song lyrics below to predict the genre")

# ---------------------------------------
# Load trained models
# ---------------------------------------
@st.cache_resource
def load_models():
    word_tfidf = pickle.load(open("word_tfidf.pkl", "rb"))
    char_tfidf = pickle.load(open("char_tfidf.pkl", "rb"))
    svm_model = pickle.load(open("svm_model.pkl", "rb"))
    return word_tfidf, char_tfidf, svm_model

word_tfidf, char_tfidf, svm_model = load_models()

# ---------------------------------------
# Cleaning (must match backend)
# ---------------------------------------
def clean_lyrics(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# ---------------------------------------
# User input
# ---------------------------------------
lyrics = st.text_area(
    "🎶 Enter song lyrics",
    height=300,
    placeholder="Paste lyrics here..."
)

# ---------------------------------------
# Prediction
# ---------------------------------------
if st.button("Predict Genre"):
    if lyrics.strip() == "":
        st.warning("Please enter some lyrics")
    else:
        with st.spinner("Analyzing lyrics..."):
            clean_text = clean_lyrics(lyrics)

            Xw = word_tfidf.transform([clean_text])
            Xc = char_tfidf.transform([clean_text])
            X = hstack([Xw, Xc])

            pred = svm_model.predict(X)[0]

        st.success(f"🎧 Predicted Genre: **{pred}**")
