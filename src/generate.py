"""Continue a prompt and list the most probable next words."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.device import get_device
from src.lstm_lm import LSTMLM
from src.vocab import Vocabulary, tokenize_prompt


def load_checkpoint(path: str | Path, device: torch.device | None = None):
    device = device or get_device()
    saved = torch.load(path, map_location=device, weights_only=False)
    vocab = Vocabulary(saved["itos"])
    config = saved["config"]
    if saved["model_type"] != "lstm":
        raise ValueError(f"Unsupported checkpoint type: {saved['model_type']}")
    model = LSTMLM(
        len(vocab),
        embed_size=config["embed_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    model.load_state_dict(saved["state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, config


def _prompt_ids(vocab: Vocabulary, prompt: str) -> list[int]:
    ids = vocab.encode(tokenize_prompt(prompt))
    if not ids:
        return [vocab.unk_id]
    return ids


def next_logits(model, ids: list[int], device: torch.device) -> torch.Tensor:
    tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(1)
    logits, _state = model(tokens)
    return logits[-1, 0]


def sample_id(logits: torch.Tensor, temperature: float) -> int:
    """Temperature 0 picks the single most likely word. Higher values sample more freely."""
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


def next_word_probs(model, vocab: Vocabulary, prompt: str, k: int = 10, device=None):
    """Top-k next words for the prompt exactly as typed, before any generation."""
    device = device or next(model.parameters()).device
    logits = next_logits(model, _prompt_ids(vocab, prompt), device)
    probabilities = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probabilities, k)
    rows = []
    for probability, index in zip(values.tolist(), indices.tolist()):
        rows.append({"word": vocab.itos[index], "probability": round(probability, 6)})
    return rows


def generate(
    model,
    vocab: Vocabulary,
    prompt: str,
    n_words: int,
    temperature: float,
    device=None,
) -> str:
    """Append n_words tokens. The model feeds its own picks back in as context."""
    device = device or next(model.parameters()).device
    ids = _prompt_ids(vocab, prompt)
    chosen: list[int] = []
    with torch.no_grad():
        for _ in range(n_words):
            logits = next_logits(model, ids, device)
            nxt = sample_id(logits, temperature)
            ids.append(nxt)
            chosen.append(nxt)
    return vocab.decode(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a continuation from a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="The history of science")
    parser.add_argument("--words", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    device = get_device()
    model, vocab, _config = load_checkpoint(args.checkpoint, device)
    print(generate(model, vocab, args.prompt, args.words, args.temperature, device))
    print()
    for row in next_word_probs(model, vocab, args.prompt, args.topk, device):
        print(f"{row['word']}\t{row['probability']:.4f}")


if __name__ == "__main__":
    main()
