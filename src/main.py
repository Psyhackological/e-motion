"""Main module to run the Gradio interface for sentiment analysis."""

from gradio_interface import demo

if __name__ == "__main__":
    demo.launch(inbrowser=True)
