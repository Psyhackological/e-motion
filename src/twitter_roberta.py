"""Module for handling Twitter RoBERTa model loading and sentiment prediction."""

import numpy as np
from scipy.special import softmax
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# Load tokenizer and model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)


def preprocess(text: str) -> str:
    """Preprocess the input text by replacing user mentions and URLs."""
    return " ".join(
        [
            "@user" if t.startswith("@") else "http" if t.startswith("http") else t
            for t in text.split()
        ],
    )


def predict_sentiment(text: str) -> dict:
    """Predict the sentiment of the given text using the RoBERTa model."""
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)[::-1]
    return {config.id2label[rank]: np.round(float(scores[rank]), 4) for rank in ranking}
