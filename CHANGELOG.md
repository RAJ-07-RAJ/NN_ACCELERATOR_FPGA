# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- (placeholder for upcoming features)

### Changed
- (placeholder)

### Fixed
- (placeholder)

---

## [1.2.0] - 2026-05-30

### Added
- Production-quality repository restructuring (industry-standard layout).
- 11 documentation files under `docs/` (architecture, verification, debug,
  bring-up, coding guidelines, FAQ, limitations, future work, project summary,
  performance, Vivado setup).
- `DESIGN_DECISIONS.md` and `PROJECT_ROADMAP.md` at project root.
- `.github/` templates: bug report, feature request, PR template, CI workflow.
- Per-category RTL subdirectories (`rtl/top/`, `rtl/layers/`, `rtl/neurons/`,
  `rtl/activation/`, `rtl/memory/`, `rtl/control/`, `rtl/common/`,
  `rtl/package/`).
- Per-category TB subdirectories matching UVM-style verification layout.
- TCL automation scripts (`build_project`, `compile`, `simulate`,
  `regression`, `clean`, `generate_reports`).
- Coverage and assertion scaffolding.

### Changed
- All Markdown documentation rewritten to industry standards.

---

## [1.1.0] - 2026-05-29

### Added
- Interactive single-image runner `python/run_inference.py`.
- 30 pre-built test cases (3 per digit) under `examples/sample_run/`.
- Regression script with PASS/FAIL CSV reporting.

### Changed
- Memory file paths made relative to simulator working directory (was
  `../mem/`) for Vivado xsim compatibility.

---

## [1.0.0] - 2026-05-28

### Added
- Initial end-to-end FPGA neural network accelerator for MNIST.
- PyTorch training (FP32, 5 epochs, 97.49% test accuracy).
- Post-training INT8 quantization (97.43% accuracy, 0.06 pp drop).
- Parameterized SystemVerilog RTL:
  - 7-state main FSM
  - DMA loader (DRAM → on-chip SRAM)
  - Weight-stationary 16-PE array
  - Drain unit + PPU (bias + requant + ReLU + INT8 saturate)
  - Ping-pong activation SRAM banks
  - Memory-mapped CSR register block
- Self-checking testbench verified bit-exact against Python on multiple
  MNIST images.
- Unit testbenches for PE and PPU.
- Vivado synthesis flow.
- Generic XDC constraints.
