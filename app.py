"""Gradio interface for next-word probabilities and text continuation."""

from pathlib import Path

import gradio as gr

from src.device import get_device
from src.generate import generate, load_checkpoint, next_word_probs

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "lstm.pt"


def load_model():
    if not CHECKPOINT.exists():
        raise SystemExit(
            "No LSTM checkpoint found. Train one first:\n  python -m src.train"
        )
    model, vocab, _config = load_checkpoint(CHECKPOINT, get_device())
    return model, vocab


MODEL, VOCAB = load_model()


def predict(prompt: str, n_words: int, temperature: float):
    continuation = generate(MODEL, VOCAB, prompt, int(n_words), float(temperature))
    rows = [[row["word"], row["probability"]] for row in next_word_probs(MODEL, VOCAB, prompt, k=10)]
    return continuation, rows


def build():
    with gr.Blocks(title="WikiText-2 language model") as demo:
        gr.Markdown(
            "Continue a phrase with an LSTM language model trained from scratch on WikiText-2. "
            "The table is the model's distribution for the very next word."
        )
        prompt = gr.Textbox(
            label="Starting phrase",
            value="The history of science",
        )
        with gr.Row():
            n_words = gr.Slider(10, 100, value=30, step=1, label="Words to generate")
            temperature = gr.Slider(
                0,
                1.5,
                value=0.8,
                step=0.1,
                label="Temperature (0 picks the most likely word)",
            )
        generate_button = gr.Button("Generate")
        continuation = gr.Textbox(label="Continuation", lines=8)
        table = gr.Dataframe(
            headers=["word", "probability"],
            datatype=["str", "number"],
            type="array",
            label="Most probable next words",
        )
        generate_button.click(
            predict,
            inputs=[prompt, n_words, temperature],
            outputs=[continuation, table],
        )
    return demo


if __name__ == "__main__":
    build().launch()
