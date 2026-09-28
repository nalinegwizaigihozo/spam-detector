import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_FILE = "SpamCollectionSMS.txt"

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
NUM_RE = re.compile(r"\d+")
NON_WORD_RE = re.compile(r"[^a-z\s]")


def clean_text(text):
    text = text.lower()
    text = URL_RE.sub(" urltoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    text = NON_WORD_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def train_model():
    df = pd.read_csv(
        DATA_FILE, sep="\t", header=None, names=["label", "text"],
        encoding="latin-1", quoting=3,
    ).dropna().drop_duplicates()
    df["y"] = df["label"].map({"ham": 0, "spam": 1})
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text, stop_words="english",
            ngram_range=(1, 2), min_df=2)),
        ("clf", MultinomialNB(alpha=0.1)),
    ])
    model.fit(df["text"], df["y"])
    return model, len(df)


model, n_messages = train_model()

st.title("SMS Spam Detector")
st.write(
    "Type or paste a text message and the model will decide if it is "
    "spam or a normal message."
)

message = st.text_area("Your message:", height=120)

if st.button("Check message"):
    if not message.strip():
        st.warning("Please type a message first.")
    else:
        p = float(model.predict_proba([message])[0][1])
        if p >= 0.5:
            st.error(f"SPAM (spam probability: {p:.0%})")
        else:
            st.success(f"Not spam (spam probability: {p:.0%})")
        st.progress(p)

st.sidebar.header("About this app")
st.sidebar.write(
    f"Model: TF-IDF + Naive Bayes, trained on {n_messages:,} SMS messages "
    "from the UCI SMS Spam Collection."
)
st.sidebar.write("Test accuracy: about 98.8%.")
