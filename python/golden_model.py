"""
golden_model.py
===============
Pure-Python bit-exact reference for the RTL accelerator.

This module exposes a single function `infer(image_int8) -> logits_int8`
that produces *exactly* the same 10 INT8 output bytes that the RTL DUT
produces for the same input. Used by:
  - generate_test_vectors.py  (to produce golden_output.mem)
  - run_inference.py          (to compare RTL vs reference)
  - accuracy_checker.py       (to evaluate on full MNIST test set)

The arithmetic mirrors `rtl/activation/ppu.sv`:
    acc = W @ x + b                          (INT32)
    acc = (acc * M_q) >> 16                  (signed arithmetic shift)
    if ReLU: acc = max(acc, 0)
    output = clip(acc, -128, 127)            (INT8 saturate)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

from model import MLP
from quantization import quant_layer, quantize_int8

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


class GoldenModel:
    """Bit-exact INT8 reference model. Construct once, infer many."""

    def __init__(self,
                 ckpt_path: str | Path = None,
                 qparams_path: str | Path = None):
        ckpt_path    = Path(ckpt_path or HERE / "results" / "mlp_mnist_best.pt")
        qparams_path = Path(qparams_path or REPO / "mem" / "quant_params.json")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        if not qparams_path.exists():
            raise FileNotFoundError(f"quant params not found: {qparams_path}")

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m  = MLP(); m.load_state_dict(ck["model_state"]); m.eval()
        qp = json.load(open(qparams_path))

        self.s_a  = [L["s_a"]  for L in qp["layers"]]
        self.s_w  = [L["s_w"]  for L in qp["layers"]]
        self.M_q  = [L["M_q"]  for L in qp["layers"]]
        self.relu = [L["relu"] for L in qp["layers"]]

        linears = [m.fc1, m.fc2, m.fc3]
        self.W_q, self.b_q = [], []
        for i, lin in enumerate(linears):
            W = lin.weight.detach().numpy(); b = lin.bias.detach().numpy()
            self.W_q.append(np.clip(np.round(W / self.s_w[i]), -128, 127).astype(np.int8))
            self.b_q.append(np.clip(np.round(b / (self.s_a[i] * self.s_w[i])),
                                    -2**31, 2**31 - 1).astype(np.int32))

    def quantize_input(self, x_norm: np.ndarray) -> np.ndarray:
        """Normalized FP32 image → INT8 input vector."""
        return quantize_int8(x_norm.flatten(), self.s_a[0])

    def infer(self, x_q: np.ndarray) -> np.ndarray:
        """INT8 input → 10 INT8 logits (bit-exact with RTL)."""
        if x_q.dtype != np.int8:
            raise TypeError("infer() expects an INT8 input vector")
        a = quant_layer(x_q,    self.W_q[0], self.b_q[0], self.M_q[0], relu=self.relu[0])
        a = quant_layer(a,      self.W_q[1], self.b_q[1], self.M_q[1], relu=self.relu[1])
        return quant_layer(a,   self.W_q[2], self.b_q[2], self.M_q[2], relu=self.relu[2])

    def predict(self, x_norm: np.ndarray) -> int:
        """FP32 image → predicted digit (argmax)."""
        x_q = self.quantize_input(x_norm)
        logits = self.infer(x_q)
        return int(np.argmax(logits.astype(np.int32)))


# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(HERE))
    from dataset import get_loaders

    p = argparse.ArgumentParser(description="Run the golden model on one MNIST image.")
    p.add_argument("--idx", type=int, default=0)
    args = p.parse_args()

    _, tl = get_loaders(batch_size=1)
    samp_x, samp_y = tl.dataset[args.idx]
    g = GoldenModel()
    pred = g.predict(samp_x[0].numpy())
    print(f"MNIST idx={args.idx}  true={int(samp_y)}  golden_pred={pred}")
