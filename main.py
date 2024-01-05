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
theme = gr.themes.Base(
    primary_hue="indigo",
    font=[gr.themes.GoogleFont("MD Mono"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Lato"), "ui-monospace", "Consolas", "monospace"],
).set(
    body_background_fill_dark="linear-gradient(45deg, rgba(23,19,57,1) 0%, rgba(6,2,13,1) 100%);",
    body_background_fill="linear-gradient(45deg, rgba(184,201,255,1) 0%, rgba(114,52,224,1) 100%);",
    body_text_color="*primary_900",
    body_text_color_subdued="*neutral_950",
    body_text_color_subdued_dark="*primary_300",
    button_secondary_background_fill="*primary_300",
    button_secondary_background_fill_dark="*primary_600",
    button_secondary_background_fill_hover="*primary_100",
    button_secondary_background_fill_hover_dark="*primary_400",
    button_secondary_text_color="*neutral_950",
)

# Gradio interface

with gr.Blocks(theme=theme, title="🙂 E-motion 🙃") as demo:
    with gr.Row():
        gr.Image(
            "assets/logos/e-motion_logo_17.svg",
            height=128,
            show_download_button=False,
            container=False,
        )
    with gr.Row():
        with gr.Column():
            box = gr.Textbox(
                placeholder="Type something to check sentiment! 🤔 ",
                label="🚀 Give it a go!",
                info="We are classifying meaning behind your text.",
                max_lines=10,
            )
            gr.ClearButton(box)
        with gr.Column():
            outputs = gr.Label(
                value="😴 nothing to show yet...",
                num_top_classes=3,
                label="results",
            )
            btn = gr.Button("Classify")
            btn.click(predict_sentiment, inputs=[box], outputs=[outputs])
    gr.Markdown(
        """
    Choose some ideas from below and see what it brings you back:
    """
    )
    gr.Examples(
        [
            "I love you.",
            "Do you wanna go eat something with us?",
            "Go away!",
            "Amazing work, I see some improvements to make though.",
            "Are you out of your mind!?",
            "I can't shake off this constant feeling of worry and fear. It's affecting my daily life, and I don't know how to cope.",
            "I can't help but feel like I'm not good enough. No matter what I do, it feels like I'm always falling short.",
            "I'm so tired of feeling like this. I just want to feel normal again.",
            "I feel like I'm going crazy. I can't stop thinking about all the things that could go wrong.",
        ],
        inputs=[box],
    )

# Launch the interface
demo.launch(inbrowser=True)
