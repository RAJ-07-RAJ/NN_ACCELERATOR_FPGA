"""
export_biases.py
================
Thin wrapper around quantization.py that emphasizes "export biases only".

In this project, biases are exported as part of the unified `export_weights.py`
run (because they share quantization scales with weights). This script exists
as a discoverability shim — invoking it just runs the unified exporter and
points the user at the bias-related output files.

Usage:
    python export_biases.py
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt",       default=str(HERE / "results" / "mlp_mnist_best.pt"))
    p.add_argument("--mem_dir",    default=str(HERE.parent / "mem"))
    p.add_argument("--sample_idx", type=int, default=0)
    args = p.parse_args()

    cmd = [sys.executable, str(HERE / "export_weights.py"),
           "--ckpt", args.ckpt,
           "--mem_dir", args.mem_dir,
           "--sample_idx", str(args.sample_idx)]
    print("[export_biases] delegating to:", " ".join(cmd))
    subprocess.check_call(cmd)

    print()
    print("[export_biases] Bias-related output files:")
    for f in ["biases/bias.mem", "biases/bias_packed.mem"]:
        p = Path(args.mem_dir) / f
        if p.exists():
            print(f"   ✓ {p}  ({p.stat().st_size:,} bytes)")
        else:
            print(f"   ✗ {p}  (missing!)")


if __name__ == "__main__":
    main()
