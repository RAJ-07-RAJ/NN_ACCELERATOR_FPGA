"""
train.py
========
Train the 784-128-64-10 MLP on MNIST.

Hyperparameter rationale (why each value):
- batch_size=128 : sweet spot for SGD noise vs throughput on a single GPU.
                   Smaller -> noisier gradients but better generalization.
                   Larger  -> faster epochs but worse final accuracy on MNIST.
- lr=1e-3        : default for Adam on small dense nets.
- optimizer=Adam : robust on MNIST; SGD+momentum also works but needs more
                   manual LR tuning. Adam converges in <10 epochs reliably.
- weight_decay=1e-4 : small L2 to keep weights bounded (helps INT8 quant later).
- scheduler=CosineAnnealingLR : smooth LR decay, no manual milestones.
- epochs=15      : MNIST saturates ~98.3% with this MLP after ~10 epochs.
- dropout=0.2    : light regularization; >0.3 starts hurting accuracy here.

Tradeoffs explained:
- Accuracy vs HW cost : adding a 256-wide hidden layer buys ~0.2% but doubles
                        DSP/BRAM. We accept the smaller net.
- FP32 vs INT8        : INT8 halves BRAM (vs INT16) and quadruples DSP packing
                        on Xilinx (DSP48E2 packs two INT8 MACs). Expected
                        accuracy loss after PTQ is < 0.5%.
- ReLU placement      : after bias-add of hidden layers only -> lets us treat
                        each layer as (MAC -> +bias -> ReLU -> requantize),
                        which is exactly the PPU pipeline in RTL.
- Batch size on HW    : the FPGA runs batch=1 inference. Training batch size
                        only affects model quality, not hardware.
"""
from __future__ import annotations
import argparse
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from model import MLP
from dataset import get_loaders


def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += crit(logits, y).item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    return correct / total, loss_sum / total, np.concatenate(all_preds), np.concatenate(all_labels)


def train_one_epoch(model, loader, opt, device):
    model.train()
    crit = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total, loss_sum / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--out_dir", type=str, default="../results")
    p.add_argument("--ckpt", type=str, default="../results/mlp_mnist_best.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device = {device}")

    train_loader, test_loader = get_loaders(batch_size=args.batch_size)
    model = MLP(dropout=args.dropout).to(device)

    opt   = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = {"train_acc": [], "train_loss": [], "test_acc": [], "test_loss": []}
    best_acc, best_path = 0.0, args.ckpt

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_acc, tr_loss = train_one_epoch(model, train_loader, opt, device)
        te_acc, te_loss, _, _ = evaluate(model, test_loader, device)
        sched.step()
        dt = time.time() - t0
        history["train_acc"].append(tr_acc); history["train_loss"].append(tr_loss)
        history["test_acc"].append(te_acc);  history["test_loss"].append(te_loss)
        print(f"epoch {epoch:02d}/{args.epochs}  "
              f"train {tr_acc*100:.2f}% / {tr_loss:.4f}   "
              f"test  {te_acc*100:.2f}% / {te_loss:.4f}   "
              f"lr={opt.param_groups[0]['lr']:.2e}   {dt:.1f}s")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({
                "model_state": model.state_dict(),
                "test_acc": te_acc,
                "args": vars(args),
            }, best_path)
            print(f"  [*] saved new best to {best_path} ({te_acc*100:.2f}%)")

    # -------- plots --------
    epochs = np.arange(1, args.epochs + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["test_loss"],  label="test")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a*100 for a in history["train_acc"]], label="train")
    plt.plot(epochs, [a*100 for a in history["test_acc"]],  label="test")
    plt.xlabel("epoch"); plt.ylabel("accuracy %"); plt.legend(); plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "training_curves.png"), dpi=140)
    plt.close()

    # Confusion matrix on best model
    ck = torch.load(best_path, map_location=device)
    model.load_state_dict(ck["model_state"])
    _, _, preds, labels = evaluate(model, test_loader, device)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (acc={best_acc*100:.2f}%)")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=7)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=140)
    plt.close()

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"[done] best test accuracy = {best_acc*100:.2f}%   ckpt -> {best_path}")


if __name__ == "__main__":
    main()
