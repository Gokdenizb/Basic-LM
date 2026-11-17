"""Perplexity for a trained model and an add-k bigram baseline."""

from __future__ import annotations

import argparse
import math
from collections import Counter

import torch
from torch import nn

from src.data import batchify, load_corpus
from src.device import get_device
from src.generate import load_checkpoint
from src.train import run_epoch


def bigram_perplexity(
    train_ids: torch.Tensor, eval_ids: torch.Tensor, vocab_size: int, k: float = 1.0
) -> float:
    """Add-k smoothed bigram perplexity. The first token uses a unigram.

    P(w | prev) = (count(prev, w) + k) / (count(prev) + k * vocab_size).
    """
    train = train_ids.tolist()
    unigram = Counter(train)
    bigram = Counter(zip(train, train[1:]))
    total = len(train)

    nll = 0.0
    seen = 0
    previous = None
    for word in eval_ids.tolist():
        if previous is None:
            probability = (unigram[word] + k) / (total + k * vocab_size)
        else:
            probability = (bigram[(previous, word)] + k) / (unigram[previous] + k * vocab_size)
        nll -= math.log(probability)
        seen += 1
        previous = word
    return math.exp(nll / seen)


def evaluate(checkpoint: str, min_freq: int, batch_size: int) -> dict[str, float]:
    device = get_device()
    model, vocab, config = load_checkpoint(checkpoint, device)
    corpus = load_corpus(min_freq)
    if corpus["itos"] != vocab.itos:
        raise SystemExit("Checkpoint vocabulary does not match the prepared corpus.")

    criterion = nn.CrossEntropyLoss()
    seq_len = config["seq_len"]
    scores = {}
    for name in ("validation", "test"):
        source = batchify(corpus["splits"][name], batch_size)
        loss = run_epoch(model, source, seq_len, criterion, device)
        scores[name] = math.exp(loss)
        print(f"{config['model_type']} {name} perplexity: {scores[name]:.2f}")

    scores["bigram_validation"] = bigram_perplexity(
        corpus["splits"]["train"], corpus["splits"]["validation"], len(vocab)
    )
    scores["bigram_test"] = bigram_perplexity(
        corpus["splits"]["train"], corpus["splits"]["test"], len(vocab)
    )
    print(f"bigram validation perplexity: {scores['bigram_validation']:.2f}")
    print(f"bigram test perplexity: {scores['bigram_test']:.2f}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and the bigram baseline.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.min_freq, args.batch_size)


if __name__ == "__main__":
    main()
