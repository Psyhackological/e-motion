from gradio_interface import setup_interface

if __name__ == "__main__":
    demo = setup_interface()
    demo.launch(inbrowser=True)
