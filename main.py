"""
This module creates a Gradio interface for sentiment analysis using the
cardiffnlp/twitter-roberta-base-sentiment-latest model.
"""

import gradio as gr
import numpy as np
from scipy.special import softmax
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)

# Load tokenizer and model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)  # Ensure config is loaded


def preprocess(text):
    """
    Preprocess the input text by replacing user mentions and URLs.

    Args:
    text (str): The text to preprocess.

    Returns:
    str: The preprocessed text.
    """
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def predict_sentiment(text):
    """
    Predict the sentiment of the given text using a RoBERTa model.

    Args:
    text (str): The text for sentiment analysis.

    Returns:
    dict: A dictionary of sentiment labels and their corresponding scores.
    """
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)[::-1]
    results = {}
    for i in range(scores.shape[0]):
        label = config.id2label[ranking[i]]
        score = np.round(float(scores[ranking[i]]), 4)
        results[label] = score
    return results


# Create Gradio Interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs="text",
    outputs="json",
)

# Launch the interface
iface.launch(inbrowser=True)
