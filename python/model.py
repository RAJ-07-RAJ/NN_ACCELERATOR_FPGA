"""
model.py
========
Fully-connected MNIST classifier: 784 -> 128 -> 64 -> 10.

Design notes (rationale):
- Two hidden layers + ReLU produce a smooth accuracy/HW-cost knee for MNIST.
  Larger nets give diminishing returns but cost more BRAM/DSP on the FPGA.
- ReLU is placed AFTER bias-add of each hidden layer (standard). It is NOT
  applied on the logits (output layer), because the hardware emits raw scores
  and argmax is computed by the host or by a tiny comparator chain.
- No batchnorm: it complicates fixed-point folding and adds runtime divides.
  Using only Linear + ReLU + Dropout keeps the quantization story clean
  (per-layer affine + activation, easy to fuse into INT8).
- Dropout is used only during training; it disappears at inference, so it
  costs nothing on the FPGA.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class MLP(nn.Module):
    """784-128-64-10 MLP, ReLU activations, optional dropout.

    The module is intentionally minimal so the post-training quantization
    flow can fuse each (Linear + ReLU) block into one INT8 layer with a
    single output scale.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden1: int = 128,
        hidden2: int = 64,
        output_size: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size

        # Layers are kept as separate attributes (not nn.Sequential) so the
        # exporter can iterate them by name and dump weights/biases in order.
        self.fc1 = nn.Linear(input_size, hidden1, bias=True)
        self.fc2 = nn.Linear(hidden1, hidden2, bias=True)
        self.fc3 = nn.Linear(hidden2, output_size, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28) or (B, 784)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)            # raw logits, no ReLU here
        return x

    # Convenience: list (name, Linear) in execution order — used by exporter.
    def linear_layers(self):
        return [("fc1", self.fc1), ("fc2", self.fc2), ("fc3", self.fc3)]
