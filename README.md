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

**e-motion** is a sentiment analysis tool, leveraging the power of AI to interpret emotions from text. This project utilizes the ROBERTa model, specifically the [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), to analyze sentiment with high accuracy. This tool is ideal for researchers, developers and people that are not sure what the sentiment of the messages is.

## Features

- **Advanced Sentiment Analysis**: Utilizes the ROBERTa model for precise sentiment prediction, with a special knack for detecting the most subtle 'meh' in a sea of text.
- **User-Friendly Interface**: Features a Gradio interface for easy interaction.
- **Gradio Integration for Sharing**: Features a Gradio-based interface, allowing for easy sharing and collaboration.
- **English Language Support**: Tailored specifically for English text, ensuring focused and effective analysis.

## Dependencies

### [Python](https://www.python.org/)
Serves as the foundational programming language, enabling the integration and execution of various libraries necessary for the project.

### [Gradio](https://pypi.org/project/gradio/)
Utilized to create a user-friendly interface for the sentiment analysis application. It simplifies the process of setting up input and output fields, facilitating the interaction between end-users and the AI model.

### [Transformers](https://pypi.org/project/transformers/)
Provides access to pre-trained models, such as RoBERTa, which is crucial for sentiment analysis. This library enables the efficient loading and utilization of the [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) model for analyzing text sentiment.

### [SciPy](https://pypi.org/project/SciPy/)
SciPy, particularly its `softmax` function, is used for converting the model's output scores into probability distributions. This step is vital in interpreting the model's predictions as meaningful probabilities.

### [NumPy](https://pypi.org/project/numpy/)
Essential for handling numerical computations and data manipulation. In this project, Numpy is used for operations such as sorting the sentiment scores and rounding them off for clearer presentation in the results.

### [Torch](https://pypi.org/project/torch/)
A deep learning framework that supports the loading and execution of the RoBERTa model. It is instrumental in managing tensors, which are key components in machine learning models like RoBERTa.

## Installation

### Prerequisites
- **Git**: Ensure Git is installed on your machine. For installation instructions, visit [Git Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
- **Python**: Ensure that Python (version 3.10 or later) is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

### Using Git Clone and Requirements.txt

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/Psyhackological/e-motion.git
   cd e-motion
   ```

2. **Create a Virtual Environment**:
   Before installing dependencies, it's recommended to create a virtual environment named `e-motion`:
   ```bash
   python3 -m venv e-motion
   source e-motion/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   With the virtual environment activated, install the necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Using Poetry

1. **Install pip and pipx**:
   - Install pip, Python's package installer, by following the instructions on [pip.pypa.io](https://pip.pypa.io/en/stable/installing/).
   - Install pipx, a tool for installing and running Python applications in isolated environments, using:
     ```bash
     python3 -m pip install --user pipx
     python3 -m pipx ensurepath
     ```

2. **Install Poetry**:
   Use pipx to install Poetry, a tool for dependency management and packaging in Python:
   ```bash
   pipx install poetry
   ```

3. **Clone the Repository**:
   ```bash
   git clone https://github.com/Psyhackological/e-motion.git
   cd e-motion
   ```

4. **Install Dependencies with Poetry**:
   Run the following command within the repository directory:
   ```bash
   poetry install
   ```

5. **Run the Application**:
   Launch the application using:
   ```bash
   poetry run python3 main.py
   ```
