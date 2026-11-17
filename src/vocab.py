"""Tokenization and a vocabulary built from the training split only."""

from __future__ import annotations

import re
from collections import Counter

UNK = "<unk>"
EOS = "<eos>"

# WikiText writes some punctuation as placeholders so it survives Moses tokenization.
PLACEHOLDERS = (("@-@", "-"), ("@,@", ","), ("@.@", "."))
# Punctuation that should sit against the previous word when we print text.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?%)\]}])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[{])\s+")
_CONTRACTION = re.compile(r"\s+((?:n't)|(?:'(?:s|re|ve|ll|d|m|t))\b)", re.IGNORECASE)


def is_heading(line: str) -> bool:
    """WikiText wraps titles as '= Heading =' or '= = Section = ='."""
    stripped = line.strip()
    return bool(stripped) and stripped.startswith("=") and stripped.endswith("=")


def normalize(text: str) -> str:
    for placeholder, symbol in PLACEHOLDERS:
        text = text.replace(placeholder, symbol)
    return text


def tokenize_line(line: str) -> list[str]:
    """Split a WikiText line on whitespace.

    The raw release already puts spaces around punctuation. A second regex pass
    would break tokens such as ``@-@`` and ``'s``.
    """
    return normalize(line).split()


def tokenize_prompt(text: str) -> list[str]:
    """Tokenize a typed phrase into the same kinds of tokens the corpus uses.

    No automatic <eos>, so generation continues the sentence instead of starting
    a new line. Contractions become WikiText pieces: ``game's`` -> ``game 's``.
    """
    text = normalize(text.strip())
    text = text.replace("n't", " n't")
    text = re.sub(r"(?i)(?<!n)'(s|re|ve|ll|d|m|t)\b", r" '\1", text)
    text = re.sub(r"([^\w\s'])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def tokens_from_rows(rows: list[str]) -> list[str]:
    """Turn WikiText rows into a token stream, dropping blanks and article headings."""
    tokens: list[str] = []
    for row in rows:
        line = row.strip()
        if not line or is_heading(line):
            continue
        words = tokenize_line(line)
        if not words:
            continue
        tokens.extend(words)
        tokens.append(EOS)
    return tokens


def detokenize(tokens: list[str]) -> str:
    """Join tokens into readable text. <eos> becomes a line break."""
    pieces: list[str] = []
    for token in tokens:
        if token == EOS:
            pieces.append("\n")
        else:
            pieces.append(token)
    text = " ".join(pieces)
    text = _CONTRACTION.sub(r"\1", text)
    text = re.sub(r"\s+-\s+", "-", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = text.replace(" \n", "\n").replace("\n ", "\n")
    return text.strip()


class Vocabulary:
    def __init__(self, itos: list[str]):
        if UNK not in itos or EOS not in itos:
            raise ValueError("Vocabulary must contain <unk> and <eos>.")
        self.itos = list(itos)
        self.stoi = {word: index for index, word in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.unk_id
        return [self.stoi.get(token, unk) for token in tokens]

    def decode(self, ids: list[int]) -> str:
        return detokenize(self.itos[index] for index in ids)


def build_vocab(train_tokens: list[str], min_freq: int = 3) -> Vocabulary:
    """Keep words that appear at least `min_freq` times in training text.

    Validation and test words that miss this cutoff encode as <unk>.
    """
    counts = Counter(train_tokens)
    itos = [UNK, EOS]
    for word, count in counts.most_common():
        if word in (UNK, EOS):
            continue
        if count >= min_freq:
            itos.append(word)
    return Vocabulary(itos)
