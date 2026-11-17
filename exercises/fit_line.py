"""Fit y = 2x with one linear layer. A training loop, before the language model."""

import torch
from torch import nn

torch.manual_seed(0)
inputs = torch.linspace(-1, 1, 64).unsqueeze(1)
targets = 2 * inputs

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

loss = None
for _step in range(200):
    optimizer.zero_grad()
    prediction = model(inputs)
    loss = loss_fn(prediction, targets)
    loss.backward()
    optimizer.step()

print(f"final loss {loss.item():.6f}")
print(f"weight {model.weight.item():.4f}")
print(f"bias {model.bias.item():.4f}")
