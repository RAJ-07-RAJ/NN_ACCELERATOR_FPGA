# Project Roadmap

A living document describing planned features, organized by phase.
Maintainers update this quarterly; contributors are welcome to claim
unassigned items.

## Vision

Make this the **reference open-source FPGA neural-network accelerator**
that:

1. **Beginners** can use to learn RTL+ML co-design end-to-end.
2. **Practitioners** can use as a starting point for production engines.
3. **Researchers** can use to benchmark new quantization / dataflow ideas.

## Versioning Scheme

- **Major (X.0.0)** — Breaking changes to RTL interface or memory map.
- **Minor (1.X.0)** — New features, new layers, new doc sections.
- **Patch (1.0.X)** — Bug fixes, doc clarifications, test additions.

## Phase 1 — Foundation ✅ DONE (v1.0.0)

- [x] PyTorch training pipeline (MNIST MLP)
- [x] Symmetric per-tensor INT8 quantization
- [x] Memory-mapped CSR + DMA + WSRAM + ASRAM + BSRAM + OSRAM
- [x] 16-PE weight-stationary array
- [x] Bias + requantize + ReLU + saturate PPU
- [x] End-to-end self-checking testbench
- [x] Vivado synthesis flow

## Phase 2 — Verification & Reproducibility ✅ DONE (v1.1.0)

- [x] 30-image regression suite
- [x] Interactive `run_inference.py` tool
- [x] PNG previews for every test case
- [x] Vivado-friendly file path handling

## Phase 3 — Productionization ✅ DONE (v1.2.0)

- [x] Industry-standard directory layout
- [x] 11 documentation files
- [x] GitHub community files (issue/PR templates, CI)
- [x] TCL automation scripts
- [x] Verilator lint clean

## Phase 4 — Performance (v1.3.0, planned Q3 2026)

- [ ] **Inter-layer pipelining**: overlap fc2 of image N with fc3 of image N-1
- [ ] **3rd activation bank** to enable double-buffering
- [ ] **Estimated speedup**: 1.8× throughput
- [ ] **Cycle-accurate model** in Python for "what-if" analysis

## Phase 5 — Quantization Quality (v1.4.0, planned Q4 2026)

- [ ] **Per-channel quantization** (weights only)
- [ ] **Asymmetric quantization** option
- [ ] **Power-of-two rescaling** to eliminate the requant multiplier
- [ ] **Calibration with histogram clipping** (currently min/max)
- [ ] **Expected accuracy gain**: +0.2 to +0.5 pp on harder datasets

## Phase 6 — CNN Support (v2.0.0, planned 2027)

- [ ] **im2col fetch unit** to feed conv as matrix-vector
- [ ] **Line buffer** for streaming feature maps
- [ ] **Pooling unit** (max + average)
- [ ] **Tested on**: MNIST LeNet-5, CIFAR-10 ResNet-Tiny

## Phase 7 — System Integration (v2.1.0)

- [ ] **AXI4 master** DRAM port (replace behavioural model)
- [ ] **AXI4-Lite slave** for CSR (replace simple bus)
- [ ] **Zynq / Versal integration example** with bare-metal C driver
- [ ] **Linux kernel driver** + character device

## Phase 8 — Modern Verification (v2.2.0)

- [ ] **Cocotb test environment**
- [ ] **UVM-style monitors and scoreboards** (if SV-LRM compliant simulator)
- [ ] **Formal properties** for FSM correctness
- [ ] **Functional coverage** with `covergroup`

## Phase 9 — Multi-network Demonstrations

- [ ] **Reconfigurable depth/width** (load different weights, reuse hardware)
- [ ] **3-class MLP** for color-classification demo
- [ ] **Autoencoder** for compression demo
- [ ] **LeNet-5** with conv added

## Stretch Goals

- [ ] **Open-source ASIC tape-out** via OpenROAD/Skywater 130nm
- [ ] **Float16 mode** for accuracy-critical layers
- [ ] **Sparsity support** (skip zero weights)
- [ ] **Multi-board scaling** (network across two FPGAs)

## How to Pick Something to Work On

1. Look for `good first issue` and `help wanted` labels on GitHub.
2. For Phase 4+ items, comment on the tracking issue to claim it.
3. For Phase 9+ stretch goals, open an issue first to discuss scope.

## Out of Scope

Things we have explicitly decided **not** to do:

- **Training on the FPGA** — out of scope; this is an inference engine.
- **GPU comparison benchmarks** — different design point, not meaningful.
- **Proprietary IP integration** (vendor MAC blocks, encrypted memories).
- **Closed-source contributions** — this is and will remain MIT.
