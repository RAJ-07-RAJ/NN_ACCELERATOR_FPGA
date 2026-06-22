"""
inference.py
============
FP32 inference utility. Loads the trained checkpoint, runs a few MNIST
samples, prints predictions, and saves a small grid figure with the
input image + predicted/true label.
"""
from __future__ import annotations
import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import MLP
from dataset import get_loaders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="../results/mlp_mnist_best.pt")
    p.add_argument("--n", type=int, default=16, help="num samples to show")
    p.add_argument("--out", type=str, default="../results/inference_samples.png")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    model = MLP().to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    _, test_loader = get_loaders(batch_size=args.n)
    x, y = next(iter(test_loader))
    x_dev = x.to(device)
    with torch.no_grad():
        logits = model(x_dev)
        pred = logits.argmax(dim=1).cpu().numpy()

    cols = 8
    rows = (args.n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.2, rows*1.4))
    for i in range(rows*cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i < args.n:
            img = x[i, 0].numpy()
            ax.imshow(img, cmap="gray")
            color = "green" if pred[i] == y[i].item() else "red"
            ax.set_title(f"p={pred[i]} t={y[i].item()}", color=color, fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    plt.close()
    print(f"[inference] saved {args.out}")


if __name__ == "__main__":
    main()
