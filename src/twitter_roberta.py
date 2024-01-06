import numpy as np
from scipy.special import softmax
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# Load tokenizer and model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def predict_sentiment(text):
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
