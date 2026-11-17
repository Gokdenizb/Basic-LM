"""Train an LSTM language model on WikiText-2."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch import nn

from src.data import ROOT, batchify, iter_batches, load_corpus
from src.device import get_device
from src.lstm_lm import LSTMLM

LSTM_CONFIG = {
    "embed_size": 200,
    "hidden_size": 200,
    "num_layers": 2,
    "dropout": 0.2,
}


def repackage(state):
    """Detach hidden state so backprop stops at the BPTT window boundary."""
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(repackage(part) for part in state)
    return state.detach()


def run_epoch(
    model, source, seq_len, criterion, device, optimizer=None, clip=1.0, log_interval=0
) -> float:
    """Mean next-token cross-entropy. Pass an optimizer to train; omit it to evaluate."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    state = None
    step = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, targets in iter_batches(source, seq_len):
            inputs = inputs.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad()
            logits, state = model(inputs, state)
            vocab_size = logits.size(-1)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            state = repackage(state)
            count = targets.numel()
            total_loss += loss.item() * count
            total_tokens += count
            step += 1
            if training and log_interval and step % log_interval == 0:
                print(
                    f"  batch {step}  loss {loss.item():.3f}  ppl {math.exp(loss.item()):.1f}",
                    flush=True,
                )
    return total_loss / total_tokens


def save_checkpoint(path: Path, model, itos, config, val_perplexity: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": config["model_type"],
            "state_dict": model.state_dict(),
            "itos": itos,
            "config": config,
            "val_perplexity": val_perplexity,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    corpus = load_corpus(args.min_freq)
    vocab = corpus["vocab"]
    train_source = batchify(corpus["splits"]["train"], args.batch_size)
    valid_source = batchify(corpus["splits"]["validation"], args.batch_size)
    print(f"vocabulary: {len(vocab)}")
    print(f"train stream shape: {tuple(train_source.shape)}")

    model = LSTMLM(len(vocab)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    config = {
        "model_type": "lstm",
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "clip": args.clip,
        "min_freq": args.min_freq,
        "seed": args.seed,
        **LSTM_CONFIG,
    }

    best = math.inf
    save_path = ROOT / args.save
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss = run_epoch(
            model,
            train_source,
            args.seq_len,
            criterion,
            device,
            optimizer,
            args.clip,
            log_interval=200,
        )
        val_loss = run_epoch(model, valid_source, args.seq_len, criterion, device)
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - started
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train ppl {train_ppl:.2f}  val ppl {val_ppl:.2f}  "
            f"({elapsed:.0f}s)"
        )
        if val_ppl < best:
            best = val_ppl
            save_checkpoint(save_path, model, vocab.itos, config, val_ppl)
            print(f"  saved {save_path}")

    print(f"best validation perplexity: {best:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a WikiText-2 language model.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=35)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--save", type=str, default="checkpoints/lstm.pt")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
