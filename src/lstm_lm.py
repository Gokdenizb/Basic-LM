"""Word-level LSTM language model.

The network estimates P(next word | previous words). An embedding turns a word
id into a learned vector. The LSTM reads that vector together with its previous
hidden state and writes a new hidden state, which is the memory of the prefix.
A linear layer turns the hidden state into a score for every vocabulary word.
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 200,
        hidden_size: int = 200,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, tokens: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """tokens: (seq, batch) integer ids. Returns logits (seq, batch, vocab) and state."""
        embedded = self.drop(self.embed(tokens))
        hidden, state = self.lstm(embedded, state)
        logits = self.output(self.drop(hidden))
        return logits, state
