<div align="center">

# Neural Network Accelerator — MNIST Classifier on FPGA

**Parameterized, weight-stationary INT8 inference engine in SystemVerilog**

[![Verified on iverilog](https://img.shields.io/badge/iverilog-12.0%20PASS-brightgreen)](docs/Verification_Guide.md)
[![Vivado xsim](https://img.shields.io/badge/Vivado-2022.2%20PASS-brightgreen)](docs/Vivado_Setup.md)
[![Regression](https://img.shields.io/badge/regression-30%2F30%20PASS-brightgreen)](examples/sample_run/REGRESSION_RESULTS.md)
[![FP32 acc](https://img.shields.io/badge/FP32%20accuracy-97.49%25-blue)](docs/Performance_Report.md)
[![INT8 acc](https://img.shields.io/badge/INT8%20accuracy-97.43%25-blue)](docs/Performance_Report.md)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![SV-2017](https://img.shields.io/badge/SystemVerilog-2017-orange)](docs/Coding_Guidelines.md)

</div>

---

## 1. Project Overview

A **fully parameterized, weight-stationary neural-network inference
accelerator** that classifies handwritten digits from the MNIST dataset.
The network is trained in PyTorch (FP32), post-training-quantized to INT8,
and implemented as synthesizable SystemVerilog targeting Xilinx 7-series
and UltraScale+ FPGAs.

```
                   ┌──────────────────────────────┐
   MNIST image ──▶ │   784 → 128 → 64 → 10 MLP    │ ──▶  predicted digit
                   │  INT8, weight-stationary,    │
                   │  16-PE array, 100 MHz        │
                   └──────────────────────────────┘
```

- **End-to-end verified**: Python → INT8 quantize → RTL → bit-exact match
  on all 10 output logits for **30/30** MNIST test images covering every digit.
- **Reproducible**: one command builds, one command tests, deterministic
  cycle count (~14,969 cycles per inference @ 100 MHz = ~150 µs).
- **Industry-standard layout**: documentation, scripts, regression suite,
  contribution templates, and CI all included.

## 2. Architecture Diagram

```
   ┌───────────────────────────────────────────────────────────────┐
   │                        nn_accel_top (DUT)                      │
   │                                                                 │
   │   ┌────────────┐   ┌────────────┐   ┌──────────────────┐       │
   │   │ config_regs│──▶│  main_fsm  │──▶│   dma_loader     │       │
   │   │   (CSR)    │   │ (7 states) │   │ DRAM→on-chip SRAM│       │
   │   └────────────┘   └────────────┘   └────────┬─────────┘       │
   │                                              │                  │
   │           ┌──────────────────────────────────┼──────────┐       │
   │           ▼                                  ▼          ▼       │
   │   ┌──────────────┐  ┌────────────────┐  ┌──────────┐            │
   │   │   WSRAM      │  │  ASRAM A/B     │  │  BSRAM   │            │
   │   │ 128b × 6848  │  │ 8b × 784 each  │  │ 32b×202  │            │
   │   │  (weights)   │  │  (ping-pong)   │  │ (biases) │            │
   │   └──────┬───────┘  └────────┬───────┘  └────┬─────┘            │
   │          │                    │                │                 │
   │          └──────────┬─────────┘                │                 │
   │                     ▼                          │                 │
   │             ┌──────────────────┐               │                 │
   │             │  compute_layer   │               │                 │
   │             │   ┌──────────┐   │               │                 │
   │             │   │stagger   │   │               │                 │
   │             │   │ align    │   │               │                 │
   │             │   └────┬─────┘   │               │                 │
   │             │        ▼         │               │                 │
   │             │   ┌──────────┐   │               │                 │
   │             │   │ PE array │   │               │                 │
   │             │   │ N=16 MACs│   │               │                 │
   │             │   └────┬─────┘   │               │                 │
   │             │        ▼         │               │                 │
   │             │   ┌──────────┐   │               │                 │
   │             │   │  drain   │◀──┼───────────────┘                 │
   │             │   └────┬─────┘   │                                 │
   │             │        ▼         │                                 │
   │             │   ┌──────────┐   │                                 │
   │             │   │   PPU    │   │   bias + (acc·M_q)>>16          │
   │             │   │          │   │   → ReLU → INT8 saturate        │
   │             │   └────┬─────┘   │                                 │
   │             └────────┼─────────┘                                 │
   │                      ▼                                            │
   │                ┌───────────┐                                      │
   │                │  OSRAM    │  10 INT8 logits                      │
   │                │ 8b × 10   │                                      │
   │                └───────────┘                                      │
   └────────────────────────────────────────────────────────────────────┘
```

Full block-level discussion: [`docs/Architecture.md`](docs/Architecture.md).
Block diagrams: [`images/`](images/).

## 3. Features

- **End-to-end pipeline**: Python training → INT8 quantization → memory
  file export → RTL inference → self-checking testbench
- **Parameterized RTL**: change any layer size, data width, or array size
  by editing one package file
- **Bit-exact reference model**: Python `quantize_layer()` matches RTL PPU
  arithmetic exactly (verified on 30/30 test images)
- **Weight-stationary dataflow**: 16 parallel PEs, one 128-bit weight word
  per cycle, optimal DSP utilization
- **Memory-mapped CSR**: AXI-Lite-style host interface
  (CTRL / STATUS / INPUT_PTR / IRQ)
- **Comprehensive verification**: top TB + unit TBs + 30-image regression
  + corner cases + coverage hooks
- **Vivado-ready**: project-build TCL, XDC constraints, full Vivado setup guide
- **Multi-simulator**: tested on Vivado xsim, Icarus Verilog 12.0,
  Verilator-lint clean
- **Hardware bring-up ready**: detailed bring-up guide for porting to a
  Xilinx Arty / Zynq board

## 4. Design Goals

| Goal                         | Approach                                                          |
|------------------------------|-------------------------------------------------------------------|
| **Synthesizable**            | `always_ff`/`always_comb`, no latches, no `force` in design code   |
| **Parameterized**            | Single `nn_pkg.sv` for all dimensions and widths                  |
| **Reproducible**             | Fixed RNG seeds, deterministic FSM, fixed-point INT8 math         |
| **Verifiable**               | Bit-exact Python reference, 30+ canned tests, regression script   |
| **Documented**               | 11 Markdown docs + per-module headers + waveform debug guide      |
| **Portable**                 | Vendor-independent SV-2017; works on Xilinx, Intel, sim-only       |
| **Scalable**                 | Path to CNN via im2col is clearly described                       |

## 5. Directory Structure

```
nn-accelerator-fpga/
├── README.md, LICENSE, CONTRIBUTING.md, CHANGELOG.md, ...
│
├── docs/                    11 docs (architecture, verification, etc.)
│
├── rtl/                     synthesizable design sources
│   ├── package/             nn_pkg.sv (parameters + types)
│   ├── top/                 nn_accel_top.sv
│   ├── layers/              compute_layer.sv
│   ├── neurons/             pe.sv, pe_array.sv
│   ├── activation/          ppu.sv (bias+requant+ReLU+sat)
│   ├── memory/              sram_sp.sv, dma_loader.sv
│   ├── control/             main_fsm.sv, config_regs.sv
│   └── common/              stagger_unit.sv, drain_unit.sv
│
├── tb/                      verification environment
│   ├── top_tb/              tb_nn_accel_top.sv (end-to-end TB)
│   ├── tests/               tb_pe.sv, tb_ppu.sv (unit TBs)
│   ├── reference_model/     dram_model.sv + Python golden
│   ├── drivers/             stimulus drivers (UVM-style stubs)
│   ├── monitors/            output monitors
│   ├── scoreboard/          golden-vs-DUT comparator
│   ├── sequences/           test sequences
│   ├── assertions/          SVA properties
│   └── coverage/            functional coverage
│
├── python/                  training + quantization + golden model
│   ├── train_model.py
│   ├── quantization.py
│   ├── export_weights.py
│   ├── export_biases.py
│   ├── generate_test_vectors.py
│   ├── golden_model.py
│   └── accuracy_checker.py
│
├── mem/                     pre-generated memory init files
│   ├── weights/             weights.mem, weights_packed.mem, requant.mem
│   ├── biases/              bias.mem, bias_packed.mem
│   ├── images/              input.mem, input_packed.mem
│   └── expected_outputs/    golden_output.mem
│
├── scripts/                 Vivado TCL automation
│   ├── build_project.tcl    one-shot project creation
│   ├── compile.tcl          compile sources only
│   ├── simulate.tcl         run sim
│   ├── regression.tcl       loop over all canned tests
│   ├── clean.tcl            delete project artefacts
│   └── generate_reports.tcl resource + timing dumps
│
├── sim/                     simulation outputs (gitignored)
│   ├── run/  logs/  waves/  coverage/  reports/
│
├── constraints/             top.xdc (100 MHz clock, IO timing)
├── synthesis/               synthesis + utilization reports
├── images/                  PNG diagrams for docs/README
└── examples/                sample_run/ with 30 canned test cases
```

## 6. Setup Instructions

### Prerequisites

| Tool       | Version | Purpose                              |
|------------|---------|--------------------------------------|
| Python     | 3.10+   | Training + quantization              |
| PyTorch    | 2.0+    | Training                             |
| Vivado     | 2022.2+ | RTL simulation (xsim) + synthesis    |
| iverilog   | 12.0+   | (Optional) command-line simulation   |
| GTKWave    | any     | (Optional) waveform viewer           |
| Verilator  | 5.0+    | (Optional) RTL lint                  |

### Clone and install

```bash
git clone https://github.com/<your-user>/nn-accelerator-fpga.git
cd nn-accelerator-fpga
pip install -r python/requirements.txt
```

### Sanity check

```bash
# Vivado one-shot
vivado -mode batch -source scripts/build_project.tcl

# OR iverilog one-shot
mkdir -p sim/run && cd sim/run
cp ../../mem/weights/weights_packed.mem .
cp ../../mem/biases/bias_packed.mem .
cp ../../mem/images/input_packed.mem .
cp ../../mem/expected_outputs/golden_output.mem .
iverilog -g2012 -I../../rtl/package -o sim.vvp -s tb_nn_accel_top \
    ../../rtl/package/nn_pkg.sv \
    ../../rtl/memory/sram_sp.sv \
    ../../rtl/neurons/pe.sv ../../rtl/neurons/pe_array.sv \
    ../../rtl/common/stagger_unit.sv ../../rtl/common/drain_unit.sv \
    ../../rtl/activation/ppu.sv \
    ../../rtl/memory/dma_loader.sv \
    ../../rtl/control/config_regs.sv ../../rtl/control/main_fsm.sv \
    ../../rtl/layers/compute_layer.sv ../../rtl/top/nn_accel_top.sv \
    ../../tb/reference_model/dram_model.sv \
    ../../tb/top_tb/tb_nn_accel_top.sv
vvp sim.vvp
```

Expected output:

```
[TB] ***** PASS : 10/10 outputs match Python golden *****
```

## 7. Vivado Simulation Flow

The fully-scripted path:

```tcl
vivado -mode batch -source scripts/build_project.tcl
vivado -mode batch -source scripts/simulate.tcl
```

Manual GUI walkthrough: see [`docs/Vivado_Setup.md`](docs/Vivado_Setup.md).

## 8. Python Training Flow

```bash
cd python
python train_model.py           # 97.49% test acc in 5 epochs (CPU ~ 1 min)
python quantization.py          # INT8 PTQ, prints per-layer scales + accuracy
```

The trained model is saved as `python/results/mlp_mnist_best.pt`.

## 9. Weight / Bias / Test-vector Generation

```bash
cd python
python export_weights.py              # writes mem/weights/*.mem
python export_biases.py               # writes mem/biases/*.mem
python generate_test_vectors.py       # writes mem/images/*.mem + golden
```

All `.mem` files use the format documented in
[`docs/Architecture.md §6`](docs/Architecture.md) and the RTL include file
`rtl/package/nn_params_auto.svh` is regenerated automatically.

## 10. Verification Flow

```bash
# Run the full self-checking testbench (end-to-end)
vivado -mode batch -source scripts/simulate.tcl

# Run regression over all 30 canned test cases
vivado -mode batch -source scripts/regression.tcl

# Inspect waveforms
gtkwave sim/waves/nn_accel.vcd
```

Verification methodology, scoreboard description, and debug strategy:
[`docs/Verification_Guide.md`](docs/Verification_Guide.md).

## 11. Sample Waveforms

Key signals during one inference (sample idx=0, "7"):

```
            ┌─load_w──┬─load_in──┬───────── compute layers ──────────┬─DONE─
state ──────┴─────────┴──────────┴────────────────────────────────────┴──────
                                  │tile0│tile1│tile2│...│tile7│ fc2 │fc3│
out_valid  ─────────────────────────────────────────────────────────────┐
                                                                         └─pulse
out[0..9]  ─XXXX─────────────────────────────────────────────────────[stable]
```

Expected waveforms and signal groups: [`docs/Debug_Guide.md`](docs/Debug_Guide.md).

## 12. Results

| Metric                              | Value                                 |
|-------------------------------------|---------------------------------------|
| FP32 test accuracy (MNIST 10k)      | **97.49 %**                           |
| INT8 test accuracy                  | **97.43 %**  (Δ = 0.06 pp)            |
| RTL ↔ Python bit-exactness          | **30 / 30** test images PASS          |
| Inference latency (cold start)      | 14,969 cycles ≈ **149.7 µs** @ 100 MHz |
| Inference latency (warm weights)    | ~7,900 cycles ≈ 79 µs                  |
| Throughput (warm)                   | **~12,650 inferences / second**       |

Full report: [`docs/Performance_Report.md`](docs/Performance_Report.md).

## 13. Performance Metrics

| Layer | IN  | OUT | Tiles (N=16) | MAC cycles | Total cycles |
|-------|-----|-----|--------------|------------|--------------|
| fc1   | 784 | 128 | 8            | 784 × 8    | 6,440        |
| fc2   | 128 | 64  | 4            | 128 × 4    | 596          |
| fc3   | 64  | 10  | 1            | 64 × 1     | 79           |

Per-inference total = 7,115 + DMA overhead ≈ 7,900 (warm) / 14,969 (cold).

## 14. Resource Utilization (Artix-7 xc7a35t)

| Resource     | Used  | Available | Util  |
|--------------|-------|-----------|-------|
| DSP48E1      | 17    | 90        | 18.9% |
| BRAM (18Kb)  | ~28   | 100       | 28.0% |
| LUT          | ~6,000| 20,800    | 28.8% |
| FF           | ~5,000| 41,600    | 12.0% |
| **Fmax**     | **>150 MHz** (estimated)            |

Full report: [`synthesis/utilization_report.md`](synthesis/utilization_report.md).

## 15. Known Limitations

- **Inference batch = 1**: no batching support yet
- **No inter-layer pipelining**: layers execute sequentially
- **FC only**: no convolution, attention, or RNN support
- **Single-port BRAMs**: DMA and compute serialized (acceptable since
  weights are loaded once, then reused)
- **Per-tensor quantization only**: no per-channel scales

Full list with rationale: [`docs/Limitations.md`](docs/Limitations.md).

## 16. Future Improvements

- Conv2D support via im2col fetch unit
- Per-channel quantization
- Multi-batch pipelining (batch ≥ 4)
- AXI4 master DRAM interface (replace behavioural model)
- Software runtime in C (replace TB-style CSR sequencing)
- Cocotb-based UVM-style verification environment

Detailed roadmap: [`PROJECT_ROADMAP.md`](PROJECT_ROADMAP.md).

---

## Documentation Index

| Document                                                | Audience                      |
|---------------------------------------------------------|-------------------------------|
| [docs/Architecture.md](docs/Architecture.md)            | RTL designers                 |
| [docs/Verification_Guide.md](docs/Verification_Guide.md)| Verification engineers        |
| [docs/Bringup_Guide.md](docs/Bringup_Guide.md)          | Hardware bring-up team        |
| [docs/Vivado_Setup.md](docs/Vivado_Setup.md)            | New users                     |
| [docs/Performance_Report.md](docs/Performance_Report.md)| Tech leads                    |
| [docs/Coding_Guidelines.md](docs/Coding_Guidelines.md)  | Contributors                  |
| [docs/Debug_Guide.md](docs/Debug_Guide.md)              | Debug engineers               |
| [docs/Limitations.md](docs/Limitations.md)              | Project planners              |
| [docs/Future_Work.md](docs/Future_Work.md)              | PMs / leads                   |
| [docs/FAQ.md](docs/FAQ.md)                              | Everyone                      |
| [docs/Project_Summary.md](docs/Project_Summary.md)      | Hiring managers / recruiters  |
| [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)              | RTL/algorithm reviewers       |
| [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md)                | Contributors                  |

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{nn_accel_fpga,
  author       = {Your Name},
  title        = {Neural Network Accelerator — MNIST Classifier on FPGA},
  year         = {2026},
  howpublished = {\url{https://github.com/<user>/nn-accelerator-fpga}},
  note         = {Parameterized INT8 weight-stationary inference engine in SystemVerilog}
}
```
