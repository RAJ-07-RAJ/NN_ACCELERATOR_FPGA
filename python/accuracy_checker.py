"""
accuracy_checker.py
===================
Evaluate the bit-exact INT8 golden model on the full MNIST test set
(or a subset) and report accuracy + per-digit confusion matrix.

Usage:
    python accuracy_checker.py                 # full 10k test set
    python accuracy_checker.py --n_eval 1000   # quick check
    python accuracy_checker.py --compare_fp32  # also report FP32 baseline
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import MLP
from dataset import get_loaders
from golden_model import GoldenModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt",         default=str(HERE / "results" / "mlp_mnist_best.pt"))
    p.add_argument("--n_eval",       type=int, default=10000,
                   help="number of test images (max 10000)")
    p.add_argument("--compare_fp32", action="store_true",
                   help="also evaluate the underlying FP32 PyTorch model")
    args = p.parse_args()

    _, test_loader = get_loaders(batch_size=512)
    g = GoldenModel(ckpt_path=args.ckpt)

    # FP32 baseline
    fp_correct = 0
    if args.compare_fp32:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        m  = MLP(); m.load_state_dict(ck["model_state"]); m.eval()
    int8_correct = 0
    total = 0

    cm = np.zeros((10, 10), dtype=np.int64)   # cm[true][pred]

    with torch.no_grad():
        for x, y in test_loader:
            x_np = x.view(x.size(0), -1).numpy()
            y_np = y.numpy()

            if args.compare_fp32:
                fp_pred = m(x).argmax(1).numpy()
                fp_correct += int((fp_pred == y_np).sum())

            for i in range(x_np.shape[0]):
                x_q = g.quantize_input(x_np[i])
                lg  = g.infer(x_q)
                pred = int(np.argmax(lg.astype(np.int32)))
                t    = int(y_np[i])
                if pred == t:
                    int8_correct += 1
                cm[t, pred] += 1
                total += 1
                if total >= args.n_eval:
                    break
            if total >= args.n_eval:
                break

    int8_acc = int8_correct / total
    print("=" * 60)
    print(f" Accuracy on {total} MNIST test images")
    print("=" * 60)
    if args.compare_fp32:
        fp_acc = fp_correct / total
        print(f"   FP32 accuracy : {fp_acc*100:.2f} %   ({fp_correct}/{total})")
        print(f"   INT8 accuracy : {int8_acc*100:.2f} %   ({int8_correct}/{total})")
        print(f"   Drop          : {(fp_acc-int8_acc)*100:+.2f} pp")
    else:
        print(f"   INT8 accuracy : {int8_acc*100:.2f} %   ({int8_correct}/{total})")

    # Per-digit accuracy
    print()
    print(" Per-digit accuracy:")
    for d in range(10):
        row_total = cm[d].sum()
        correct   = cm[d, d]
        if row_total > 0:
            print(f"   digit {d}: {correct}/{row_total}  ({100*correct/row_total:5.2f} %)")

    # Confusion matrix
    print()
    print(" Confusion matrix (rows=true, cols=predicted):")
    print("       " + "".join(f"{i:>5}" for i in range(10)))
    for i in range(10):
        print(f"   {i:>2} |" + "".join(f"{cm[i, j]:>5}" for j in range(10)))


if __name__ == "__main__":
    main()
