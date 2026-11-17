"""Download WikiText-2 and cut it into BPTT batches.

Each batch column is a contiguous stretch of the corpus. Targets are the same
tokens shifted one position forward, which is the next-word prediction task.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.vocab import Vocabulary, build_vocab, tokens_from_rows

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def cache_path(min_freq: int) -> Path:
    return DATA_DIR / f"wikitext2_min{min_freq}.pt"


def load_corpus(min_freq: int = 3) -> dict:
    """Return vocabulary and integer tensors for train, validation, and test.

    The processed file is cached under data/ so later runs skip the download.
    """
    path = cache_path(min_freq)
    if path.exists():
        saved = torch.load(path, weights_only=False)
        saved["vocab"] = Vocabulary(saved["itos"])
        return saved

    from datasets import load_dataset

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_tokens = tokens_from_rows(dataset["train"]["text"])
    vocab = build_vocab(train_tokens, min_freq=min_freq)

    splits = {"train": torch.tensor(vocab.encode(train_tokens), dtype=torch.long)}
    raw_tokens = {"train": train_tokens}
    for name in ("validation", "test"):
        tokens = tokens_from_rows(dataset[name]["text"])
        raw_tokens[name] = tokens
        splits[name] = torch.tensor(vocab.encode(tokens), dtype=torch.long)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"itos": vocab.itos, "splits": splits, "min_freq": min_freq}, path)
    return {
        "itos": vocab.itos,
        "vocab": vocab,
        "splits": splits,
        "min_freq": min_freq,
        "raw_tokens": raw_tokens,
    }


def batchify(ids: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Arrange a 1-D token stream into shape (length, batch_size).

    Leftover tokens that do not fill a complete row of streams are dropped.
    """
    n_batches = ids.size(0) // batch_size
    trimmed = ids[: n_batches * batch_size]
    return trimmed.view(batch_size, -1).t().contiguous()


def iter_batches(source: torch.Tensor, seq_len: int):
    """Yield (input, target) windows of at most `seq_len` steps.

    Both tensors have shape (window, batch_size). target[t] is the token that
    follows input[t] in the same stream.
    """
    length = source.size(0)
    for start in range(0, length - 1, seq_len):
        window = min(seq_len, length - 1 - start)
        inputs = source[start : start + window]
        targets = source[start + 1 : start + 1 + window]
        yield inputs, targets


def describe(min_freq: int = 3, batch_size: int = 20, seq_len: int = 35) -> None:
    corpus = load_corpus(min_freq)
    vocab: Vocabulary = corpus["vocab"]
    splits = corpus["splits"]
    print(f"vocabulary size: {len(vocab)}")
    print(f"min frequency: {corpus['min_freq']}")
    for name, ids in splits.items():
        unk = int((ids == vocab.unk_id).sum())
        print(f"{name}: {ids.numel()} tokens, {unk} <unk> ({unk / ids.numel():.2%})")

    train_ids = splits["train"]
    counts = {}
    for index in train_ids.tolist():
        token = vocab.itos[index]
        if token == "<eos>":
            continue
        if token == "<unk>":
            continue
        counts[token] = counts.get(token, 0) + 1
    top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:20]
    print("20 most common training words:")
    for word, count in top:
        print(f"  {word}\t{count}")

    source = batchify(train_ids, batch_size)
    inputs, targets = next(iter_batches(source, seq_len))
    print(f"input shape: {tuple(inputs.shape)}")
    print(f"target shape: {tuple(targets.shape)}")
    if not torch.equal(inputs[1:, 0], targets[:-1, 0]):
        raise SystemExit("Target is not the input shifted by one token.")
    print("decoded input window:")
    print(vocab.decode(inputs[:, 0].tolist()))
    print("decoded target window:")
    print(vocab.decode(targets[:, 0].tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WikiText-2 and print sanity checks.")
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=35)
    args = parser.parse_args()
    describe(args.min_freq, args.batch_size, args.seq_len)


if __name__ == "__main__":
    main()
