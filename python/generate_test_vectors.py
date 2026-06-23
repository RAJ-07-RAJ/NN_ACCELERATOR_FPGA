"""
generate_test_vectors.py
========================
Generate a batch of test inputs (MNIST images) + matching golden outputs.

Produces, for each picked image:
    examples/sample_run/test_inputs/digit_<D>/idx_<I>/input.mem
    examples/sample_run/test_inputs/digit_<D>/idx_<I>/input_packed.mem
    examples/sample_run/test_inputs/digit_<D>/idx_<I>/golden_output.mem
    examples/sample_run/test_inputs/digit_<D>/idx_<I>/preview.png
    examples/sample_run/test_inputs/digit_<D>/idx_<I>/info.txt

Plus a top-level INDEX.md and manifest.csv.

Usage:
    python generate_test_vectors.py                       # default: 3 per digit
    python generate_test_vectors.py --per_digit 5         # 50 cases
    python generate_test_vectors.py --indices 0 5 42 100  # specific indices
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from model import MLP
from dataset import get_loaders, MNIST_MEAN, MNIST_STD
from quantization import quant_layer, quantize_int8, to_hex


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt",      default=str(HERE / "results" / "mlp_mnist_best.pt"))
    p.add_argument("--qparams",   default=str(REPO / "mem" / "quant_params.json"))
    p.add_argument("--out_dir",   default=str(REPO / "examples" / "sample_run" / "test_inputs"))
    p.add_argument("--per_digit", type=int, default=3)
    p.add_argument("--indices",   type=int, nargs="*", default=None)
    args = p.parse_args()

    # --- load model + quant params ---------------------------------------
    if not Path(args.ckpt).exists():
        sys.exit(f"[error] checkpoint not found: {args.ckpt}\n"
                 f"        Train first: python train_model.py")
    if not Path(args.qparams).exists():
        sys.exit(f"[error] quant params not found: {args.qparams}\n"
                 f"        Run once: python export_weights.py")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = MLP(); model.load_state_dict(ck["model_state"]); model.eval()
    qp = json.load(open(args.qparams))

    layers = [model.fc1, model.fc2, model.fc3]
    s_a   = [L["s_a"]  for L in qp["layers"]]
    s_w   = [L["s_w"]  for L in qp["layers"]]
    M_q   = [L["M_q"]  for L in qp["layers"]]
    relu  = [L["relu"] for L in qp["layers"]]
    W_q, b_q = [], []
    for i, lin in enumerate(layers):
        W = lin.weight.detach().numpy(); b = lin.bias.detach().numpy()
        W_q.append(np.clip(np.round(W / s_w[i]), -128, 127).astype(np.int8))
        b_q.append(np.clip(np.round(b / (s_a[i] * s_w[i])),
                           -2**31, 2**31 - 1).astype(np.int32))

    # --- pick which MNIST indices to use ---------------------------------
    _, test_loader = get_loaders(batch_size=1)
    ds = test_loader.dataset

    if args.indices is not None:
        picks = list(args.indices)
    else:
        by_d = defaultdict(list)
        for i in range(len(ds)):
            _, y = ds[i]
            y = int(y)
            if len(by_d[y]) < args.per_digit:
                by_d[y].append(i)
            if all(len(v) >= args.per_digit for v in by_d.values()) and len(by_d) == 10:
                break
        picks = [i for d in range(10) for i in by_d[d]]

    print(f"[gen] generating {len(picks)} test cases under {args.out_dir}")
    out_root = Path(args.out_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    manifest = ["digit,idx,python_prediction,folder\n"]
    index_md = ["# Test Input Catalog\n\n",
                f"{len(picks)} ready-to-use test cases.\n\n",
                "| Digit | Test idx | Folder | Python predicts | Correct? |\n",
                "|-------|----------|--------|-----------------|----------|\n"]

    for idx in picks:
        x_t, y_t = ds[idx]
        true_label = int(y_t)
        x_norm = x_t[0].numpy()

        # quantize input + run bit-exact INT8 pipeline
        x_q = quantize_int8(x_norm.flatten(), s_a[0])
        a = x_q
        a = quant_layer(a, W_q[0], b_q[0], M_q[0], relu=relu[0])
        a = quant_layer(a, W_q[1], b_q[1], M_q[1], relu=relu[1])
        logits = quant_layer(a, W_q[2], b_q[2], M_q[2], relu=relu[2])
        py_pred = int(np.argmax(logits.astype(np.int32)))

        case = out_root / f"digit_{true_label}" / f"idx_{idx:04d}"
        case.mkdir(parents=True, exist_ok=True)

        # write the four files
        with open(case / "input.mem", "w") as f:
            f.write(f"// INT8 input image, test idx={idx}, label={true_label}\n")
            for v in x_q: f.write(to_hex(int(v), 8) + "\n")

        with open(case / "input_packed.mem", "w") as f:
            f.write("// input, INT8 in LSBs, zero-extended\n")
            for v in x_q: f.write(f"{int(v) & 0xff:032x}\n")

        with open(case / "golden_output.mem", "w") as f:
            f.write(f"// INT8 golden output for test idx={idx}, true label={true_label}, q-pred={py_pred}\n")
            for v in logits: f.write(to_hex(int(v), 8) + "\n")

        # preview PNG
        img = np.clip(x_norm * MNIST_STD[0] + MNIST_MEAN[0], 0, 1)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        color = "green" if py_pred == true_label else "red"
        ax.set_title(f"idx={idx}  true={true_label}  py={py_pred}", color=color, fontsize=10)
        ax.axis("off"); plt.tight_layout()
        plt.savefig(case / "preview.png", dpi=140); plt.close()

        # info.txt
        with open(case / "info.txt", "w") as f:
            f.write(f"MNIST test set idx : {idx}\n")
            f.write(f"True label         : {true_label}\n")
            f.write(f"Python prediction  : {py_pred}\n")
            f.write(f"Correct?           : {'YES' if py_pred==true_label else 'NO'}\n")
            f.write(f"INT8 logits        : {list(map(int, logits))}\n")

        ok = "OK" if py_pred == true_label else "model wrong"
        rel = f"digit_{true_label}/idx_{idx:04d}"
        manifest.append(f"{true_label},{idx},{py_pred},{rel}\n")
        index_md.append(f"| {true_label} | {idx} | `{rel}` | {py_pred} | {ok} |\n")

    (out_root / "manifest.csv").write_text("".join(manifest))
    (out_root / "INDEX.md").write_text("".join(index_md))

    print(f"[gen] done. {len(picks)} cases at {out_root}")


if __name__ == "__main__":
    main()
