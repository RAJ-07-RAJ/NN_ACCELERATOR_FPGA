"""
run_inference.py
================
Interactive end-to-end "pick an image -> feed to RTL -> verify" tool.

WHAT THIS DOES
--------------
1. Loads the trained MLP checkpoint.
2. Lets you pick an MNIST test image (by --idx, or --random, or your own PNG).
3. Displays the image (saves PNG so you can see what you fed in).
4. Quantizes it to INT8 and writes the .mem files the RTL needs.
5. Runs the Python INT8 reference and prints the predicted digit.
6. Optionally runs the iverilog RTL simulation and compares.
7. Prints a clean side-by-side report: TRUE / PYTHON / RTL.

USAGE
-----
  # pick a specific MNIST test image
  python run_inference.py --idx 42

  # pick a random one
  python run_inference.py --random

  # supply your own 28x28 grayscale PNG
  python run_inference.py --image my_digit.png

  # only run Python, skip the RTL simulation
  python run_inference.py --idx 7 --no-rtl

  # run the RTL simulation too (default if iverilog is installed)
  python run_inference.py --idx 7 --rtl
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import MLP
from dataset import get_loaders, MNIST_MEAN, MNIST_STD
from quantization import quant_layer, quantize_int8, to_hex

HERE      = Path(__file__).resolve().parent
PROJECT   = HERE.parent
MEM_DIR   = PROJECT / "mem"
SCRIPTS   = PROJECT / "scripts"
SIM_DIR   = PROJECT / "sim"
RESULTS   = PROJECT / "results"
CKPT      = RESULTS / "mlp_mnist_best.pt"


# ============================================================================
# helpers
# ============================================================================
def load_model_and_quant():
    """Load the trained checkpoint and the per-layer quantisation parameters."""
    if not CKPT.exists():
        sys.exit(f"[error] checkpoint not found: {CKPT}\n"
                 f"        run `cd python && python train.py` first.")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = MLP()
    model.load_state_dict(ck["model_state"])
    model.eval()

    qp_file = MEM_DIR / "quant_params.json"
    if not qp_file.exists():
        sys.exit(f"[error] quant params not found: {qp_file}\n"
                 f"        run `cd python && python export_weights.py` once.")
    qp = json.load(open(qp_file))
    return model, qp, ck.get("test_acc", None)


def load_mnist_image(idx: int | None,
                     random_pick: bool,
                     png_path: str | None):
    """Return (x_float_28x28, true_label_int) for the chosen image."""
    if png_path is not None:
        from PIL import Image
        img = Image.open(png_path).convert("L").resize((28, 28))
        arr = np.array(img, dtype=np.float32) / 255.0
        # Apply the same normalisation as training (mean/std)
        arr = (arr - MNIST_MEAN[0]) / MNIST_STD[0]
        return arr, None, f"PNG {png_path}"

    _, test_loader = get_loaders(batch_size=1)
    ds = test_loader.dataset
    if random_pick:
        idx = int(np.random.randint(0, len(ds)))
    elif idx is None:
        idx = 0
    if not 0 <= idx < len(ds):
        sys.exit(f"[error] idx {idx} out of range 0..{len(ds)-1}")

    x_tensor, y_int = ds[idx]              # x_tensor: (1, 28, 28) normalised
    arr = x_tensor[0].numpy()              # (28, 28) float, already normalised
    return arr, int(y_int), f"MNIST test idx={idx}"


def visualize(x_norm: np.ndarray, label: int | None, src: str, out_png: Path):
    """Save a PNG showing the input image so the user can see what was fed in."""
    # de-normalise just for display
    img_display = x_norm * MNIST_STD[0] + MNIST_MEAN[0]
    img_display = np.clip(img_display, 0, 1)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img_display, cmap="gray", vmin=0, vmax=1)
    title = src + (f"   true label = {label}" if label is not None else "")
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def print_ascii_image(x_norm: np.ndarray):
    """Print the image to the terminal so the user sees it without a viewer."""
    img = x_norm * MNIST_STD[0] + MNIST_MEAN[0]
    img = np.clip(img, 0, 1)
    chars = " .:-=+*#%@"
    print("\n[image preview (terminal)]:")
    for row in img:
        line = "".join(chars[int(min(len(chars)-1, p * len(chars)))] for p in row)
        print("    " + line)
    print()


# ============================================================================
# quantize + write .mem files
# ============================================================================
def quantize_and_dump(model: MLP, qp: dict, x_norm: np.ndarray, true_label: int | None):
    """Quantize the chosen image, run the Python INT8 pipeline, and write
       input.mem + golden_output.mem so the RTL TB consumes the same data."""

    layers = [model.fc1, model.fc2, model.fc3]
    s_a  = [L["s_a"]  for L in qp["layers"]]
    s_w  = [L["s_w"]  for L in qp["layers"]]
    M_q  = [L["M_q"]  for L in qp["layers"]]
    relu = [L["relu"] for L in qp["layers"]]

    W_q = []
    b_q = []
    for i, lin in enumerate(layers):
        W = lin.weight.detach().numpy()
        b = lin.bias.detach().numpy()
        W_q.append(np.clip(np.round(W / s_w[i]), -128, 127).astype(np.int8))
        b_q.append(np.clip(np.round(b / (s_a[i] * s_w[i])),
                           -2**31, 2**31 - 1).astype(np.int32))

    # input -> INT8
    x_q = quantize_int8(x_norm.flatten(), s_a[0])

    # forward through the INT8 pipeline (bit-exact w/ RTL)
    a = x_q
    a = quant_layer(a, W_q[0], b_q[0], M_q[0], relu=relu[0])
    a = quant_layer(a, W_q[1], b_q[1], M_q[1], relu=relu[1])
    logits = quant_layer(a, W_q[2], b_q[2], M_q[2], relu=relu[2])
    py_pred = int(np.argmax(logits.astype(np.int32)))

    # write input.mem
    with open(MEM_DIR / "input.mem", "w") as f:
        f.write(f"// INT8 input image. true_label={true_label}\n")
        for v in x_q:
            f.write(to_hex(int(v), 8) + "\n")

    # write golden_output.mem
    with open(MEM_DIR / "golden_output.mem", "w") as f:
        f.write(f"// INT8 golden output. py_pred={py_pred}, true={true_label}\n")
        for v in logits:
            f.write(to_hex(int(v), 8) + "\n")

    return logits, py_pred


# ============================================================================
# memory packing + iverilog simulation
# ============================================================================
def pack_mem():
    subprocess.check_call([sys.executable, str(SCRIPTS / "pack_mem.py")],
                          cwd=str(SCRIPTS),
                          stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def run_iverilog():
    """Returns (success, rtl_logits[10] or None, rtl_argmax or None, cycles or None, full_log)."""
    if shutil.which("iverilog") is None:
        return False, None, None, None, "iverilog not installed (skipping RTL run)"

    try:
        proc = subprocess.run(["make", "-s", "sim"], cwd=str(SIM_DIR),
                              capture_output=True, text=True, timeout=300)
        log = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return False, None, None, None, "[error] iverilog simulation timed out"

    # Parse the TB output
    rtl_logits = []
    rtl_arg = None
    cycles  = None
    for ln in log.splitlines():
        # Lines like "[TB]   3  |     05  /    5    |     05  /    5     |       OK"
        if ln.startswith("[TB]") and "|" in ln and "/" in ln and "GOLDEN" not in ln:
            try:
                parts = [p.strip() for p in ln.split("|")]
                # parts ~ ["[TB]   3", "05  /    5", "05  /    5", "OK"]
                rtl_dec = int(parts[1].split("/")[1].strip())
                rtl_logits.append(rtl_dec)
            except Exception:
                pass
        if "RTL    argmax" in ln:
            try: rtl_arg = int(ln.split("=")[-1].strip())
            except: pass
        if "cycle_count register" in ln:
            try: cycles = int(ln.split("=")[1].split()[0])
            except: pass

    success = "PASS" in log and len(rtl_logits) == 10
    return success, rtl_logits if rtl_logits else None, rtl_arg, cycles, log


# ============================================================================
# main
# ============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Pick an MNIST image -> quantize -> RTL -> verify.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--idx",    type=int, help="MNIST test-set index (0..9999)")
    g.add_argument("--random", action="store_true", help="pick a random test image")
    g.add_argument("--image",  type=str, help="path to a 28x28 grayscale PNG")

    p.add_argument("--no-rtl", dest="rtl", action="store_false",
                   help="skip the iverilog simulation (Python only)")
    p.add_argument("--rtl", dest="rtl", action="store_true",
                   help="run the iverilog RTL simulation (default if available)")
    p.set_defaults(rtl=True)

    p.add_argument("--no-ascii", action="store_true",
                   help="don't print the ASCII art preview")
    p.add_argument("--show-rtl-log", action="store_true",
                   help="print the full RTL simulator log")
    args = p.parse_args()

    MEM_DIR.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    # 1) load model + the (already-calibrated) quantisation parameters
    print("=" * 70)
    print(" STEP 1: load trained model + quant params")
    print("=" * 70)
    model, qp, fp_acc = load_model_and_quant()
    print(f"  checkpoint        : {CKPT.name}")
    if fp_acc is not None:
        print(f"  saved FP32 acc    : {fp_acc*100:.2f}%")
    print(f"  network           : 784 -> 128 -> 64 -> 10")
    print(f"  INT8 frac bits    : {qp['frac_bits']}")
    for L in qp["layers"]:
        print(f"    {L['name']}: s_w={L['s_w']:.4g}  s_a={L['s_a']:.4g}  "
              f"M_q={L['M_q']:>5}  relu={L['relu']}")

    # 2) pick the image
    print()
    print("=" * 70)
    print(" STEP 2: pick input image")
    print("=" * 70)
    x_norm, true_label, src = load_mnist_image(args.idx, args.random, args.image)
    print(f"  source            : {src}")
    print(f"  true label        : {true_label}")
    preview_png = RESULTS / "last_input.png"
    visualize(x_norm, true_label, src, preview_png)
    print(f"  preview saved     : {preview_png}")
    if not args.no_ascii:
        print_ascii_image(x_norm)

    # 3) quantize + write .mem
    print("=" * 70)
    print(" STEP 3: quantize image + write .mem files for RTL")
    print("=" * 70)
    py_logits, py_pred = quantize_and_dump(model, qp, x_norm, true_label)
    print(f"  wrote             : mem/input.mem, mem/golden_output.mem")
    print(f"  Python INT8 logits: {list(map(int, py_logits))}")
    print(f"  Python prediction : {py_pred}")

    # 4) pack mem
    print()
    print("=" * 70)
    print(" STEP 4: pack memory files for the DRAM model")
    print("=" * 70)
    pack_mem()
    print(f"  wrote             : mem/input_packed.mem  + weights/bias packed")

    # 5) run RTL
    rtl_logits = rtl_arg = cycles = None
    rtl_ok = False
    if args.rtl:
        print()
        print("=" * 70)
        print(" STEP 5: simulate the RTL accelerator on Icarus Verilog")
        print("=" * 70)
        rtl_ok, rtl_logits, rtl_arg, cycles, rtl_log = run_iverilog()
        if rtl_ok:
            print(f"  iverilog          : PASS")
            print(f"  RTL logits        : {rtl_logits}")
            print(f"  RTL prediction    : {rtl_arg}")
            print(f"  RTL cycles        : {cycles}  (~{cycles/100:.1f} us @100 MHz)")
        else:
            print("  iverilog          : NOT RUN or FAILED")
            print("  " + rtl_log.splitlines()[-1] if rtl_log else "")
        if args.show_rtl_log:
            print("\n----- iverilog full log -----")
            print(rtl_log)

    # 6) verify
    print()
    print("=" * 70)
    print(" STEP 6: verification report")
    print("=" * 70)
    print()
    header = f"   {'neuron':>6}  {'Python':>8}  " + (f"{'RTL':>8}  match" if rtl_logits else "")
    print(header)
    print("   " + "-" * (len(header) - 3))
    mismatches = 0
    for i in range(10):
        line = f"   {i:>6d}  {int(py_logits[i]):>8d}  "
        if rtl_logits is not None:
            ok = (int(py_logits[i]) == rtl_logits[i])
            line += f"{rtl_logits[i]:>8d}  " + (" OK " if ok else "MISMATCH")
            if not ok: mismatches += 1
        marker = "  <-- argmax" if i == py_pred else ""
        print(line + marker)

    print()
    print(f"   true label        : {true_label}")
    print(f"   Python predicted  : {py_pred}  "
          + ("[CORRECT]" if true_label is not None and py_pred == true_label
             else ("[WRONG]" if true_label is not None else "")))
    if rtl_arg is not None:
        print(f"   RTL    predicted  : {rtl_arg}  "
              + ("[CORRECT]" if true_label is not None and rtl_arg == true_label
                 else ("[WRONG]"   if true_label is not None else "")))
        if rtl_arg == py_pred and mismatches == 0:
            print("\n   *** RTL <-> PYTHON : BIT-EXACT MATCH on all 10 logits ***")
        elif rtl_arg == py_pred:
            print(f"\n   ** argmax matches but {mismatches} logits differ **")
        else:
            print(f"\n   !! RTL prediction differs from Python !!")
    print()


if __name__ == "__main__":
    main()
