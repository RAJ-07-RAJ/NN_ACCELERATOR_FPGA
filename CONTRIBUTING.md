# Contributing

Thanks for your interest in contributing! This document describes the
expectations, workflow, and quality bar for changes to this project.

## Code of Conduct

Be respectful and constructive. We accept contributions from engineers
of all experience levels.

## Quick Start for Contributors

```bash
git clone https://github.com/<your-user>/nn-accelerator-fpga.git
cd nn-accelerator-fpga
pip install -r python/requirements.txt
sudo apt-get install iverilog gtkwave   # Ubuntu/Debian

# Verify everything works before making changes
bash scripts/quick_check.sh             # or vivado -mode batch -source scripts/simulate.tcl
```

If you see `***** PASS : 10/10 outputs match Python golden *****`, you're
good to start hacking.

## Ways to Contribute

| Type            | Examples                                                      |
|-----------------|---------------------------------------------------------------|
| 🐛 Bug fixes    | Fix mis-rounding in PPU, off-by-one in FSM, etc.              |
| ✨ Features     | Per-channel quantization, AXI4 master, Conv2D layer           |
| 📚 Documentation | Improve diagrams, fix typos, add examples                    |
| ✅ Verification  | Add corner-case tests, coverpoints, assertions                |
| ⚡ Performance   | Pipeline optimizations, latency reductions                    |
| 🔧 Tooling      | Better TCL scripts, CI workflow, Verilator support            |

## Workflow

1. **Open an Issue** describing the bug/feature *before* writing code,
   especially for non-trivial changes.
2. **Fork** the repository.
3. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feat/per-channel-quant
   ```
4. **Make your changes**, following the [coding guidelines](docs/Coding_Guidelines.md).
5. **Run the regression suite**:
   ```bash
   vivado -mode batch -source scripts/regression.tcl
   ```
   All 30 cases must still PASS.
6. **Update documentation** under `docs/` and the `CHANGELOG.md`.
7. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat(ppu): add per-channel requant multiplier support
   fix(compute_layer): correct off-by-one in bias address pipeline
   docs(verification): add waveform debug strategy
   test(pe): add stream-length stress tests
   refactor(memory): simplify DMA address generator
   ```
8. **Open a Pull Request** using the provided template
   (`.github/PULL_REQUEST_TEMPLATE.md`).

## Coding Standards (Summary)

Full doc: [`docs/Coding_Guidelines.md`](docs/Coding_Guidelines.md).

### SystemVerilog (RTL)

- SystemVerilog-2017.
- `always_ff @(posedge clk or negedge rst_n)` for sequential.
- `always_comb` for combinational. Initialize all defaults to avoid latches.
- Non-blocking (`<=`) in FF blocks, blocking (`=`) in combo blocks.
- `UPPER_SNAKE_CASE` parameters, `lower_snake_case` signals.
- Shared constants live in `rtl/package/nn_pkg.sv` — do not duplicate them.

### Python

- Python 3.10+.
- PEP 8, type hints where helpful.
- All randomness must be seeded.
- Use `argparse` for CLI tools with `--help` and sensible defaults.

### Documentation

- Markdown in `docs/`.
- Update `docs/` for every architectural change.
- Block diagrams in ASCII OR PNG (under `images/`).

## Pull Request Checklist

Confirm before opening the PR:

- [ ] All 30 regression test cases PASS
- [ ] No new Verilator lint warnings
- [ ] `docs/` updated if behavior changed
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Conventional commit messages
- [ ] No `force` statements in `rtl/` (TB only)
- [ ] No absolute file paths
- [ ] No accidental commit of `*.mem` regeneration if not intended

## Reporting Bugs

Use the *Bug Report* issue template
(`.github/ISSUE_TEMPLATE/bug_report.md`). Include:

- Vivado / iverilog version
- Python version, PyTorch version
- Exact command you ran
- Expected vs actual output
- Minimum reproducer (preferably an MNIST test index)

## Proposing Features

Open a *Feature Request* issue using the provided template, with a brief
design rationale. Wait for triage before opening a PR for non-trivial work.

## Maintainers

| Role              | Contact                         |
|-------------------|---------------------------------|
| Project lead      | @<github-handle>                |
| RTL maintainer    | @<github-handle>                |
| Verification lead | @<github-handle>                |

## Recognition

All contributors are listed in `CHANGELOG.md` next to the release that
includes their first contribution.

Thank you! ⚡
