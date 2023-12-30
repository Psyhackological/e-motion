<div align="center">

# :slightly_smiling_face: e-motion :upside_down_face:

![ProjectLogo](assets/e-motion_logo.png)

"Being happy, being sad - with this tool you become sentiment giga chad!"

[![License](https://img.shields.io/badge/License-MIT-black.svg?style=flat-square&color=171339)](LICENSE)
![PythonVersionRequirement](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square&color=171339)
![Version](https://img.shields.io/badge/version-0.1.0-blue?style=flat-square&color=171339)


![DemoGif](assets/demo.gif)

</div>

## :books: Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Dependencies](#dependencies)
   - [gradio](#gradio)
   - [transformers](#transformers)
   - [torch](#torch)
   - [numpy](#numpy)
   - [scipy](#scipy)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Code Structure](#code-structure)
   - [preprocess](#preprocess)
   - [predict_sentiment](#predict-sentiment)
7. [ROBERTa model](#roberta-model)
8. [Credits](#Credits)
9. [License](#license)

## Introduction

**e-motion** is a sentiment analysis tool, leveraging the power of AI to interpret emotions from text. This project utilizes the ROBERTa model, specifically the [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), to analyze sentiment with high accuracy. This tool is ideal for researchers, developers and people that are not sure what the sentiment of the messages is.

## Features

- **Advanced Sentiment Analysis**: Utilizes the ROBERTa model for precise sentiment prediction, with a special knack for detecting the most subtle 'meh' in a sea of text.
- **User-Friendly Interface**: Features a Gradio interface for easy interaction.
- **Gradio Integration for Sharing**: Features a Gradio-based interface, allowing for easy sharing and collaboration.
- **English Language Support**: Tailored specifically for English text, ensuring focused and effective analysis.
