<div align="center">

# :smile: e-motion :upside_down_face:

![ProjectLogo](assets/e-motion_logo.png)

"Being happy, being sad - with this tool you become sentiment giga chad!"

[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
![PythonVersionRequirement](https://img.shields.io/badge/python-3.10+-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-blue)


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

**e-motion** is a sentiment analysis tool, leveraging the power of AI to interpret emotions from text. This project utilizes the ROBERTa model, specifically the `cardiffnlp/twitter-roberta-base-sentiment-latest`, to analyze sentiment with high accuracy. This tool is ideal for researchers, developers, and businesses looking to gain insights from textual data.

## Features

- **Advanced Sentiment Analysis**: Utilizes the ROBERTa model for precise sentiment prediction, with a special knack for detecting the most subtle 'meh' in a sea of text.
- **User-Friendly Interface**: Features a Gradio interface for easy interaction.
- **Gradio Integration for Sharing**: Features a Gradio-based interface, allowing for easy sharing and collaboration.
- **English Language Support**: Tailored specifically for English text, ensuring focused and effective analysis.

## Dependencies

Listed dependencies include Python (>=3.10, <3.13), Gradio, Transformers, Scipy, Numpy, and Torch. Detailed versions are specified in the `pyproject.toml` file.

## Installation

1. **Clone the Repository**: Clone or download this repository to your local machine.
2. **Install Dependencies**: Run `pip install -r requirements.txt` to install the necessary libraries.
3. **Set Up Poetry**: Ensure Poetry is installed for dependency management.

## Usage

To use **e-motion**, run the main.py file. This will launch the Gradio interface in your browser. Simply input the text you wish to analyze, and the system will return the sentiment analysis results in JSON format.

## Code Structure

- **main.py**: The core module, creating the Gradio interface and defining the sentiment analysis model and functions.
- **preprocess Function**: Handles text preprocessing, replacing user mentions and URLs.
- **predict_sentiment Function**: Core function to predict sentiment from text.

## ROBERTa Model

e-motion uses the `cardiffnlp/twitter-roberta-base-sentiment-latest` model, renowned for its performance in sentiment analysis tasks. WIP: WRITE MORE

## Credits

https://www.analyticsvidhya.com/blog/2022/10/a-gentle-introduction-to-roberta/
https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest?text=I+can%27t+concentrate+on+anything%2C+and+my+thoughts+are+always+clouded+with+sadness
https://www.geeksforgeeks.org/overview-of-roberta-model/

## License

This project is licensed under the MIT License - see the LICENSE file for details.