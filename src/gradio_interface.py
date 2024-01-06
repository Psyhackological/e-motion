"""Module for setting up the Gradio interface for sentiment analysis."""

import gradio as gr
from twitter_roberta import predict_sentiment

theme = gr.themes.Base(
    primary_hue="indigo",
    font=[
        gr.themes.GoogleFont("MD Mono"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
    font_mono=[
        gr.themes.GoogleFont("Lato"),
        "ui-monospace",
        "Consolas",
        "monospace",
    ],
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


"""Set up the Gradio interface for the application."""
with gr.Blocks(theme=theme, title="🙂 E-motion 🙃") as demo:
    with gr.Row():
        with gr.Column(scale=3):
            pass
        with gr.Column(scale=1):
            gr.Image(
                "../assets/logos/e-motion_logo_17.svg",
                height=145,
                show_download_button=False,
                container=False,
                interactive=False,
            )
        with gr.Column(scale=3):
            pass
    with gr.Row():
        with gr.Column():
            box = gr.Textbox(
                placeholder="Type something to check sentiment! 🤔",
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
            # pylint: disable=no-member
            btn.click(predict_sentiment, inputs=[box], outputs=[outputs])
            # pylint: enable=no-member
    gr.Markdown("Choose some ideas from below and see what it brings you back:")
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


if __name__ == "__main__":
    demo.launch()
