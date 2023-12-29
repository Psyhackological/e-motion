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


# gradio interface theme
custom_theme = gr.themes.Base(
    primary_hue="yellow",
    secondary_hue="cyan",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont('DM Mono'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=[gr.themes.GoogleFont('ui-monospace'), 'ui-monospace', 'Consolas', 'monospace'],
).set(
    body_background_fill_dark='repeating-linear-gradient(45deg, *background_fill_primary, *background_fill_primary 10px, *background_fill_secondary 10px, *background_fill_secondary 20px)',
    body_text_color='*neutral_900',
    body_text_color_dark='*neutral_200',
    body_text_size='*text_lg',
    body_text_color_subdued_dark='*primary_500'
)

# Gradio interface
examples=[
    "I love you.",
    "Do you wanna go eat something with us?",
    "Go away!",
    "Amazing work, I see some improvements to make though.",
    "Are you out of your mind!?"
]

inputs = gr.Textbox(
    placeholder="Input text to verify emotion!",
    label="🚀 Give it a Go!",
    info="We are classifying meaning behind your text.",
    max_lines=16,
)

outputs = gr.Label(
    value="💤😴",
    num_top_classes=3,
    label="result",
)

demo = gr.Interface(
    fn=predict_sentiment,
    title="🙂 e-motion 🙃",
    # description="Simple program helping to understand what might be the meaning behind some words.",
    article="Choose some ideas from here.",
    examples=examples,
    # description="Description of the page",
    inputs=inputs,
    outputs=outputs,
    theme=custom_theme,
)

# Launch the interface
demo.launch(inbrowser=True)
