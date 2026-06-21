"""
quantize.py
===========
Post-training quantization (PTQ) to a hardware-friendly fixed-point format.

Format chosen
-------------
- Weights      : INT8, symmetric, per-tensor (per-layer) scale.
- Activations  : INT8, symmetric, per-tensor scale, calibrated on a subset
                 of the training set.
- Biases       : INT32, scale = scale_input * scale_weight  (standard rule).
- Accumulator  : INT32 inside the PE (DATA_WIDTH=8, ACC_WIDTH=32).

For every Linear layer the math is:

    y_fp = W_fp * x_fp + b_fp                     (floating point ground truth)
    y_q  = clip( round( (W_q * x_q + b_q) * M ), -128, 127 )
    where M = (s_x * s_w) / s_y   (the "requantization multiplier")

We export `M` as a fixed-point Qm.n number (default Q0.16) so the hardware
PPU can implement requantization as `mul + arithmetic_shift_right`,
which costs one DSP and one shifter — no division, no float.

Why this matters for FPGA
-------------------------
- INT8 weights/activations  ->  ~4x BRAM savings vs FP32, ~2x vs INT16.
- INT8 MACs pack 2-per-DSP48 on Xilinx Ultrascale+ -> doubles throughput.
- Requantize-by-multiply-shift avoids dividers, which are huge on FPGA.
- A single per-layer scale (not per-channel) keeps the RTL simple: one
  16-bit multiplier + one shifter per layer at the PPU.

Per-layer scaling
-----------------
For Linear `i`:
    s_w[i] = max(|W_i|) / 127
    s_a[i] = max(|act_i|, calibration) / 127      (input act of that layer)
    s_y[i] = max(|y_i|,   calibration) / 127      (output act before ReLU)
    M[i]   = (s_a[i] * s_w[i]) / s_y[i]
    b_q[i] = round(b_fp[i] / (s_a[i] * s_w[i]))   (INT32)

Saturation / overflow
---------------------
- Accumulator is 32-bit signed. For a 784-wide dot of INT8*INT8 the worst-
  case magnitude is 784 * 127 * 127 ≈ 1.26e7, which fits comfortably in INT32
  (~2.1e9). So no intra-MAC saturation is needed.
- After requantization we hard-saturate to INT8 [-128, 127].

Outputs
-------
Writes:
    mem/weights.mem        (hex, one INT8 per line, fc1 then fc2 then fc3,
                            in row-major output-major order: see export_weights.py)
    mem/bias.mem           (hex, one INT32 per line, fc1 then fc2 then fc3)
    mem/input.mem          (hex, 784 INT8 lines — one calibrated test image)
    mem/golden_output.mem  (hex, 10 INT8 lines — quantized model's logits)
    mem/requant.mem        (hex, one Q0.16 multiplier per layer, 3 lines)
    mem/quant_params.json  (all scales + shapes, for the testbench)
"""
from __future__ import annotations
import argparse
import os
import json
import numpy as np
import torch

from model import MLP
from dataset import get_loaders


# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------
INT8_MIN, INT8_MAX = -128, 127


def sym_scale(x: np.ndarray) -> float:
    """Symmetric per-tensor scale so that max(|x|) maps to 127."""
    m = float(np.max(np.abs(x)))
    if m == 0.0:
        return 1.0
    return m / 127.0


def quantize_int8(x: np.ndarray, scale: float) -> np.ndarray:
    q = np.round(x / scale)
    q = np.clip(q, INT8_MIN, INT8_MAX)
    return q.astype(np.int8)


def quantize_int32(x: np.ndarray, scale: float) -> np.ndarray:
    q = np.round(x / scale)
    q = np.clip(q, -(2**31), 2**31 - 1)
    return q.astype(np.int32)


def to_hex(v: int, width_bits: int) -> str:
    """Two's-complement hex string of given bit width, no prefix."""
    mask = (1 << width_bits) - 1
    return f"{(int(v) & mask):0{width_bits//4}x}"


# -------------------------------------------------------------------------
# quantized inference (numpy reference, matches RTL exactly)
# -------------------------------------------------------------------------
def quant_layer(x_q: np.ndarray,
                W_q: np.ndarray, b_q: np.ndarray,
                M_q16: int, relu: bool) -> np.ndarray:
    """One quantized FC layer using exactly the RTL math:
       acc = W_q @ x_q + b_q                 (INT32)
       acc = (acc * M_q16) >> 16             (arithmetic shift, signed)
       if relu: max(acc, 0)
       saturate to INT8
    """
    acc = W_q.astype(np.int32) @ x_q.astype(np.int32) + b_q.astype(np.int32)
    # multiply-shift requantize (signed)
    acc = (acc.astype(np.int64) * np.int64(M_q16)) >> 16
    if relu:
        acc = np.maximum(acc, 0)
    acc = np.clip(acc, INT8_MIN, INT8_MAX).astype(np.int8)
    return acc


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="../results/mlp_mnist_best.pt")
    p.add_argument("--mem_dir", type=str, default="../mem")
    p.add_argument("--results_dir", type=str, default="../results")
    p.add_argument("--n_calib", type=int, default=1024)
    p.add_argument("--n_eval",  type=int, default=10000)
    p.add_argument("--frac_bits", type=int, default=16,
                   help="Fractional bits in the requant multiplier (Q0.n).")
    p.add_argument("--array_size", type=int, default=16, help="Hardware PE-array width (N) for weight tiling.")
    p.add_argument("--sample_idx", type=int, default=0,
                   help="Which test image to dump as input.mem.")
    args = p.parse_args()

    os.makedirs(args.mem_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = "cpu"  # all of this is cheap on CPU
    ck = torch.load(args.ckpt, map_location=device)
    model = MLP().to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    train_loader, test_loader = get_loaders(batch_size=512)

    # ---- 1. collect calibration activations (FP32) ----
    print("[quant] collecting calibration activations...")
    acts_in  = {0: [], 1: [], 2: []}    # input to layer i
    acts_out = {0: [], 1: [], 2: []}    # pre-activation output of layer i (before ReLU)

    seen = 0
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.view(x.size(0), -1)
            a0 = x
            z1 = model.fc1(a0);    a1 = torch.relu(z1)
            z2 = model.fc2(a1);    a2 = torch.relu(z2)
            z3 = model.fc3(a2)
            acts_in[0].append(a0.numpy());  acts_out[0].append(z1.numpy())
            acts_in[1].append(a1.numpy());  acts_out[1].append(z2.numpy())
            acts_in[2].append(a2.numpy());  acts_out[2].append(z3.numpy())
            seen += x.size(0)
            if seen >= args.n_calib:
                break
    for k in acts_in:
        acts_in[k]  = np.concatenate(acts_in[k],  axis=0)
        acts_out[k] = np.concatenate(acts_out[k], axis=0)

    # ---- 2. compute scales ----
    layers = model.linear_layers()  # [("fc1", fc1), ("fc2", fc2), ("fc3", fc3)]
    W_fp = [lin.weight.detach().numpy() for _, lin in layers]
    b_fp = [lin.bias.detach().numpy()   for _, lin in layers]

    s_w = [sym_scale(W) for W in W_fp]
    s_a = [sym_scale(acts_in[i])  for i in range(3)]
    s_y = [sym_scale(acts_out[i]) for i in range(3)]

    # Quantize tensors
    W_q = [quantize_int8(W_fp[i], s_w[i]) for i in range(3)]
    b_q = [quantize_int32(b_fp[i], s_a[i] * s_w[i]) for i in range(3)]

    # Requant multiplier M = (s_a * s_w) / s_y, encoded as Q0.frac_bits
    one = 1 << args.frac_bits
    M_float = [(s_a[i] * s_w[i]) / s_y[i] for i in range(3)]
    M_q     = [int(round(m * one)) for m in M_float]
    # Sanity: M is typically << 1 for hidden layers, so Q0.16 fits well.
    for i, m in enumerate(M_q):
        if not (0 <= m < (1 << 31)):
            print(f"[warn] layer {i}: requant multiplier {m} out of signed 31-bit range; "
                  f"increase --frac_bits or rescale.")

    print("\n[quant] per-layer scales:")
    for i, (name, _) in enumerate(layers):
        print(f"  {name}: s_w={s_w[i]:.5g}  s_a={s_a[i]:.5g}  "
              f"s_y={s_y[i]:.5g}  M={M_float[i]:.5g}  M_q={M_q[i]}")

    # ---- 3. evaluate INT8 model on the full test set ----
    print("\n[quant] evaluating quantized model on test set...")
    correct_q, correct_fp, total = 0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x_np = x.view(x.size(0), -1).numpy()
            # FP32 reference
            fp_pred = model(x).argmax(1).numpy()

            # INT8 simulation, one sample at a time (matches RTL batch=1)
            for i in range(x_np.shape[0]):
                a = quantize_int8(x_np[i], s_a[0])
                a = quant_layer(a, W_q[0], b_q[0], M_q[0], relu=True)
                a = quant_layer(a, W_q[1], b_q[1], M_q[1], relu=True)
                a = quant_layer(a, W_q[2], b_q[2], M_q[2], relu=False)
                pred_q = int(np.argmax(a.astype(np.int32)))
                if pred_q == int(y[i].item()): correct_q += 1
            correct_fp += int((fp_pred == y.numpy()).sum())
            total += y.size(0)
            if total >= args.n_eval:
                break

    fp_acc = correct_fp / total
    q_acc  = correct_q  / total
    print(f"\n  FP32 accuracy : {fp_acc*100:.2f}%")
    print(f"  INT8 accuracy : {q_acc*100:.2f}%")
    print(f"  quant loss    : {(fp_acc - q_acc)*100:+.2f} pp")

    # ---- 4. dump .mem files ----
    print(f"\n[quant] writing memory files to {args.mem_dir}/")

    # weights.mem : layout for the hardware's weight-stationary tile schedule.
    # The PE array reads ONE packed word per cycle = ARRAY_SIZE weights for the
    # current input index k, one weight per PE.  Within a layer:
    #
    #   for tile in 0 .. ceil(OUT/N) - 1:
    #     for k in 0 .. IN-1:
    #       for n in 0 .. N-1:                          # packed in one word
    #         W[tile*N + n, k]   (zero-padded if tile*N+n >= OUT)
    #
    # pack_mem.py later groups every N consecutive INT8 lines into a single
    # packed word.  IMPORTANT: each tile's row count must be padded to a
    # multiple of N so the packing aligns; we do that here.
    N = args.array_size if hasattr(args, "array_size") else 16
    with open(os.path.join(args.mem_dir, "weights.mem"), "w") as fw:
        fw.write(f"// INT8 weights, tile/input/PE-packed for ARRAY_SIZE={N}.\n")
        for li, W in enumerate(W_q):
            OUT, IN = W.shape
            num_tiles = (OUT + N - 1) // N
            fw.write(f"// ---- layer {li} : shape {W.shape}, {num_tiles} tiles of {N} ----\n")
            for tile in range(num_tiles):
                for k in range(IN):
                    for n in range(N):
                        o = tile * N + n
                        v = int(W[o, k]) if o < OUT else 0
                        fw.write(to_hex(v, 8) + "\n")

    # bias.mem : INT32, one per output neuron, fc1 fc2 fc3
    with open(os.path.join(args.mem_dir, "bias.mem"), "w") as fb:
        fb.write("// INT32 biases. fc1(128) fc2(64) fc3(10)\n")
        for li, b in enumerate(b_q):
            fb.write(f"// ---- layer {li} : shape {b.shape} ----\n")
            for v in b:
                fb.write(to_hex(int(v), 32) + "\n")

    # requant.mem : Q0.frac_bits multipliers, signed 32-bit
    with open(os.path.join(args.mem_dir, "requant.mem"), "w") as fm:
        fm.write(f"// Q0.{args.frac_bits} requant multipliers (32-bit signed)\n")
        for li, m in enumerate(M_q):
            fm.write(f"// layer {li}  M_float={M_float[li]:.6g}\n")
            fm.write(to_hex(m, 32) + "\n")

    # input.mem : one calibrated INT8 image
    sample_x, sample_y = test_loader.dataset[args.sample_idx]
    x_np = sample_x.view(-1).numpy()
    x_q  = quantize_int8(x_np, s_a[0])
    with open(os.path.join(args.mem_dir, "input.mem"), "w") as fi:
        fi.write(f"// INT8 input image, test idx={args.sample_idx}, label={int(sample_y)}\n")
        for v in x_q:
            fi.write(to_hex(int(v), 8) + "\n")

    # golden_output.mem : run the same image through the quantized pipeline
    a = x_q
    a = quant_layer(a, W_q[0], b_q[0], M_q[0], relu=True)
    a = quant_layer(a, W_q[1], b_q[1], M_q[1], relu=True)
    out = quant_layer(a, W_q[2], b_q[2], M_q[2], relu=False)
    with open(os.path.join(args.mem_dir, "golden_output.mem"), "w") as fo:
        fo.write(f"// INT8 golden output for test idx={args.sample_idx}, "
                 f"true label={int(sample_y)}, q-pred={int(np.argmax(out))}\n")
        for v in out:
            fo.write(to_hex(int(v), 8) + "\n")

    # quant_params.json : everything the testbench / docs need
    params = {
        "format": "INT8 symmetric per-tensor, INT32 bias, Q0.{} requant".format(args.frac_bits),
        "frac_bits": args.frac_bits,
        "layers": [
            {"name": "fc1", "in": 784, "out": 128, "relu": True,
             "s_w": s_w[0], "s_a": s_a[0], "s_y": s_y[0],
             "M_float": M_float[0], "M_q": M_q[0]},
            {"name": "fc2", "in": 128, "out": 64,  "relu": True,
             "s_w": s_w[1], "s_a": s_a[1], "s_y": s_y[1],
             "M_float": M_float[1], "M_q": M_q[1]},
            {"name": "fc3", "in": 64,  "out": 10,  "relu": False,
             "s_w": s_w[2], "s_a": s_a[2], "s_y": s_y[2],
             "M_float": M_float[2], "M_q": M_q[2]},
        ],
        "accuracy": {"fp32": fp_acc, "int8": q_acc, "drop_pp": fp_acc - q_acc},
        "sample": {"idx": args.sample_idx, "label": int(sample_y),
                   "q_pred": int(np.argmax(out))},
    }
    with open(os.path.join(args.mem_dir, "quant_params.json"), "w") as fj:
        json.dump(params, fj, indent=2)
    with open(os.path.join(args.results_dir, "quant_report.json"), "w") as fj:
        json.dump(params, fj, indent=2)

    print("[quant] done.")


if __name__ == "__main__":
    main()
