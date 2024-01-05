<div align="center">

![e-motion banner2.gif](assets/banners/banner2.gif)

# :slightly_smiling_face: e-motion :upside_down_face:

"Being happy, being sad - with this tool you become sentiment giga chad!"

[![License](https://img.shields.io/badge/License-MIT-black.svg?style=flat-square&color=171339)](LICENSE)
![PythonVersionRequirement](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square&color=171339)
![Version](https://img.shields.io/badge/version-0.1.0-blue?style=flat-square&color=171339)


![DemoGif](assets/demo.gif)

</div>

## :books: Table of Contents

1. [Introduction](#wave-introduction)
2. [Features](#art-features)
3. [Dependencies](#package-dependencies)
   - [gradio](#-gradio)
   - [transformers](#-transformers)
   - [torch](#torch)
   - [numpy](#numpy)
   - [scipy](#scipy)
4. [Installation](#floppy_disk-installation)
5. [Usage](#rocket-usage)
6. [Code Structure](#file_folder-code-structure)
   - [preprocess](#preprocess)
   - [predict_sentiment](#predict-sentiment)
   - [How Gradio Works](#how-gradio-works)
7. [ROBERTa model](#brain-roberta-model)
8. [Credits](#clap-credits)
9. [License](#scroll-license)

## :wave: Introduction

**e-motion** is a sentiment analysis tool, leveraging the power of AI to interpret emotions from text. This project utilizes the ROBERTa model, specifically the [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), to analyze sentiment with high accuracy. This tool is ideal for researchers, developers and people that are not sure what the sentiment of the messages is.

## :art: Features

- **Advanced Sentiment Analysis**: Utilizes the ROBERTa model for precise sentiment prediction, with a special knack for detecting the most subtle 'meh' in a sea of text.
- **User-Friendly Interface**: Features a Gradio interface for easy interaction.
- **Gradio Integration for Sharing**: Features a Gradio-based interface, allowing for easy sharing and collaboration.
- **English Language Support**: Tailored specifically for English text, ensuring focused and effective analysis.

## :package: Dependencies

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

## :floppy_disk: Installation

### Prerequisites
- **Git**: Ensure Git is installed on your machine. For installation instructions, visit [Git Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
- **Python**: Ensure that Python (version 3.10 or later) is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).
- **Python-pip**: Install pip, Python's package installer, following the instructions at [pip Installation Guide](https://pip.pypa.io/en/stable/installation/).

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
   source e-motion/bin/activate  # On Windows use `e-motion\Scripts\activate`
   ```

3. **Install Dependencies**:
   With the virtual environment activated, install the necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Using Poetry

1. **Install pipx**:
   - First, ensure pip is installed as per the prerequisites.
   - Then install pipx, a tool for installing and running Python applications in isolated environments, using:
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

## :rocket: Usage

This application can be run either directly using Poetry or within a virtual environment. After starting the application by any method, it can typically be accessed at [`http://127.0.0.1:7860`](http://127.0.0.1:7860/?__theme=dark). To close the server use `CTRL + C`.

### With Poetry

1. **Standard Run**:
   - For regular execution without auto-reloading, run:
     ```bash
     poetry run python3 main.py
     ```

2. **With Gradio Auto-Reloading**:
   - For development with [auto-reloading, using Gradio's feature](https://www.gradio.app/guides/developing-faster-with-reload-mode), run:
     ```bash
     poetry run gradio main.py
     ```

### In a Virtual Environment

1. **Entering the Virtual Environment**:
   - **Poetry's Virtual Environment**:
     ```bash
     poetry shell
     ```
   - **Virtual Environment with `requirements.txt`**:
     ```bash
     source e-motion/bin/activate  # On Windows use `e-motion\Scripts\activate`
     ```

2. **Running the Application**:
   - After entering the virtual environment, run:
     ```bash
     python3 main.py
     ```

## :file_folder: Code Structure

### [`preprocess`](main.py#L19)
The `preprocess` function is designed to prepare input text for sentiment analysis by sanitizing user mentions and URLs. This preprocessing step ensures that the sentiment analysis focuses on the text's content rather than extraneous elements like user handles or links.

- **Arguments**:
  - `text (str)`: The text to be preprocessed.
- **Returns**:
  - `str`: The sanitized text, with user mentions replaced with "@user" and URLs with "http".

### [`predict_sentiment`](main.py#L37)
The `predict_sentiment` function leverages the RoBERTa model for sentiment analysis. It first preprocesses the input text, then uses the tokenizer and model to predict sentiment scores, which are converted to probabilities using the softmax function.

- **Arguments**:
  - `text (str)`: The text for sentiment analysis.
- **Returns**:
  - `dict`: A dictionary mapping sentiment labels to their corresponding scores.

### How Gradio Works

Gradio is used to create an interactive web interface for the sentiment analysis application. It allows users to input text and receive sentiment analysis results in a user-friendly format.

- **Interface Setup**:
  - The interface is defined within a `gr.Blocks` context, allowing for a customizable layout with components like text boxes, buttons, and labels.
- **Input and Output Components**:
  - Users input text into a textbox, and the sentiment analysis results are displayed in a label.
- **Event Handling**:
  - The `predict_sentiment` function is linked to a button click, triggering sentiment analysis upon user interaction.
- **Theming**:
  - A custom theme is applied to the interface for aesthetic enhancement.
- **Additional Features**:
  - The interface includes an image banner, clear button, markdown text for instructions, and predefined example inputs for easy testing.
- **Launching the Interface**:
  - The interface is launched and made accessible in a web browser using `demo.launch(inbrowser=True)`.

## :brain: RoBERTa model

e-motion uses the `cardiffnlp/twitter-roberta-base-sentiment-latest` model, renowned for its performance in sentiment analysis tasks. Here some more information about base model - `RoBERTa`:

RoBERTa is like a super smart language program made by Meta in 2020. It's way better than its older version, `BERT` made by Google in 2018, because it learned from a much bigger collection of information.

Here's what makes RoBERTa special:

- **Big Learning Data:** RoBERTa learned from a really, really huge amount of data, much more than BERT. This helps RoBERTa `understand words and sentences better for tricky language tasks`.
- **Smart Word Guessing:** RoBERTa guesses words in a different way than BERT. `It tries to figure out words that are hidden in a sentence`. This helps it understand how words relate to each other.
- **Learning from Different Examples:** RoBERTa `learns from lots of different examples by changing words or translating sentences`. This makes it really good at handling different types of language.

RoBERTa shows off its skills in many language jobs:

- **Feeling Detector:** RoBERTa is great at figuring out if a piece of writing is `happy`, `sad`, or just `neutral`. It's better at this than BERT.
- **Question Answerer:** RoBERTa can quickly answer questions about a piece of writing. It's `faster`and `smarter` at understanding information compared to BERT.
- **Logic Expert:** RoBERTa can tell if one sentence `logically connects to another`. It's better at this logical stuff than BERT.
- **Short and Sweet Summarizer:** RoBERTa can make `short and accurate summaries of long pieces of writing`. It's better at picking out the important parts compared to BERT.
- **Language Translator:** RoBERTa can `smoothly` change words from one language to another, making translations sound `more natural`. It's better at this than BERT.

What data has been used to train RoBERTa? Here are the datasets that we found:

- **BOOK CORPUS and English Wikipedia dataset**: This data also used for training BERT architecture, this data contains `16GB` of text.
- **CC-NEWS**: This data contains 63 million English news articles crawled between September 2016 and February 2019. The size of this dataset is `76 GB` after filtering.
- **OPENWEBTEXT**: This dataset contains web content extracted from the URLs shared on Reddit with at least 3 upvotes. The size of this dataset is `38 GB`.
- **STORIES**: This dataset contains a subset of Common Crawl data filtered to match the story-like style of Winograd NLP task. This dataset contains `31 GB` of text.

Overall, we can see that it was trained with `161 GB of data`. Huuuuge amount!

The tool we're using makes RoBERTa even better by giving it special training on Twitter. This helps RoBERTa:

- **Get Twitter Language Better:** The tool has learned a lot from Twitter, including `slang`, `emojis`, and how people talk casually. This means RoBERTa can understand and respond to Twitter-style language more easily.
- **Figure Out Feelings in Tweets:** RoBERTa is fine-tuned to `know the emotions` in Tweets, like if they're `happy`, `sad`, or `angry`. This makes it good for tasks like understanding how people feel and what they think.
- **Find Important Stuff in Tweets:** RoBERTa can do `more than just understand feelings`. It can also find trends, topics people are talking about, and the overall mood of a conversation in Tweets.
- **Handle Lots of Tweets Fast:** The tool is `built to quickly deal with a bunch of Tweets`. This makes it great for checking what's happening on Twitter in real-time.

**RoBERTa is like a language superhero and we are excited about what else RoBERTa can do in the future – that’s why we gave it a shot!**

## :clap: Credits

For Gradio purposes:
- [Taking first steps in Gradio](https://www.gradio.app/guides/quickstart)
- [Gradio Interface documentation](https://www.gradio.app/docs/interface)

For understanding RoBERTA model:
- [Introduction to RoBERTa](https://www.analyticsvidhya.com/blog/2022/10/a-gentle-introduction-to-roberta/)
- [Overview of RoBERTa model](https://www.geeksforgeeks.org/overview-of-roberta-model/)

For creating this super fun and pleasent project:
- [Twitter-roBERTa-base for Sentiment Analysis model from HuggingFace](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest?text=I+can%27t+concentrate+on+anything%2C+and+my+thoughts+are+always+clouded+with+sadness)

## :scroll: License

This project is licensed under the terms of the [MIT License](LICENSE).

![MIT Image](https://upload.wikimedia.org/wikipedia/commons/0/0c/MIT_logo.svg)

Made with :purple_heart: by [34Daniel](https://github.com/34panda),[D3nzer](https://github.com/D3nzer) and [me](https://github.com/Psyhackological).
