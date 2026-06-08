# Design Decisions

A record of every non-obvious architectural decision in this project,
the alternatives considered, and the rationale for the chosen path.
This is the place to look when you ask **"Why did they build it this way?"**

## Format

Each decision is captured as a lightweight Architecture Decision Record (ADR):

> **Context** — What was the situation?
> **Decision** — What did we choose?
> **Alternatives** — What else did we consider?
> **Consequences** — What are the tradeoffs?

---

## ADR-001 — Dataflow choice: Weight Stationary

**Context.** MNIST MLP layers have one large dimension (IN = 784) and one
small dimension (OUT ≤ 128). We need to pick a dataflow for the PE array.

**Decision.** Weight Stationary.

**Alternatives considered.**

| Dataflow         | Reuses                           | MNIST fit                              |
|------------------|----------------------------------|----------------------------------------|
| Weight Stationary| Weights resident in PEs          | ✅ Loads 128b/cycle, perfect DSP use   |
| Output Stationary| Accumulators stay in PEs         | OK, but needs N-deep scratch          |
| Row Stationary   | Both, via 2-D mesh (Eyeriss)     | Overkill for FC, complex routing      |
| Input Stationary | Activation stays in PE           | Wasteful: IN=784 is the big dimension |

**Consequences.**
- +Activation bandwidth is just 8b/cycle (broadcast).
- +Weight read = 1 packed word/cycle, full DSP utilization.
- −Less flexible for CONV (will need im2col adapter in Phase 6).

---

## ADR-002 — Quantization: Symmetric Per-Tensor INT8

**Context.** Need to fit weights+activations into BRAM/DSP-friendly format.

**Decision.** Symmetric (zero-point = 0) per-tensor INT8 weights and activations,
INT32 biases at scale `s_a × s_w`, Q0.16 requantize multiplier.

**Alternatives considered.**

| Scheme                | Pros                          | Cons                                |
|-----------------------|-------------------------------|-------------------------------------|
| INT8 per-tensor (✓)   | One scale per layer; simplest | Slight accuracy loss vs per-channel |
| INT8 per-channel      | Higher accuracy               | Per-output multiplier required       |
| Asymmetric            | Better fit for skewed acts    | Adds zero-point arithmetic           |
| INT16                 | Easy, lossless                | 2× BRAM, no DSP packing             |
| FP16/BF16             | No retraining loss            | DSP intensive, slow                 |

**Consequences.**
- +One 32-bit multiplier in PPU (shared across all output neurons of a layer).
- +Bit-exact reference in Python (no FP rounding mismatch).
- +Measured accuracy drop: only 0.06 pp on MNIST.
- −Future per-channel work will need PPU rearchitecture.

---

## ADR-003 — Requantization: Multiply-then-Shift, not Divide

**Context.** The mathematically correct requantize is `acc * (s_a*s_w/s_y)`.
This is a real-valued multiply that hardware can't do natively.

**Decision.** Approximate as `(acc * M_q) >> FRAC_BITS` where `M_q = round((s_a*s_w/s_y) * 2^FRAC_BITS)`. We use `FRAC_BITS = 16` (Q0.16).

**Alternatives.**
- **Floating-point**: 100x area, unnecessary precision.
- **Hardware divider**: huge area + multi-cycle latency.
- **Power-of-two scale** (shift only): faster but adds 0.5–2 pp accuracy loss.

**Consequences.**
- +One multiply + one arithmetic shift per output neuron.
- +The `M_q` value is constant per layer → baked at compile time
  (`nn_params_auto.svh`).
- +The shift-by-16 truncates rather than rounds; this matches PyTorch's
  default behavior so the Python reference matches exactly.

---

## ADR-004 — PE Pipeline Depth = 1 (combinational multiply, registered accumulator)

**Context.** Original design had a 2-stage PE (register inputs, register
product, register accumulator). This caused subtle clr/en alignment bugs.

**Decision.** Single-stage PE — multiplier combinational, only the
accumulator is registered.

**Alternatives.**
- 2-stage: easier timing closure, harder to align `clr` with first MAC.
- 3-stage: even worse alignment problem.

**Consequences.**
- +Trivial timing: INT8×INT8 multiply on a DSP48 closes 300+ MHz easily.
- +Cleaner FSM: `clr` resets the accumulator on cycle T; first MAC happens
  on T+1; the last accumulator value is valid at T+IN+1.
- +Saved a full debug cycle (the original 2-stage version had an X-propagation
  bug that took hours to track).
- −If we ever scale to INT16 or FP16, may need to re-add a pipeline stage.

---

## ADR-005 — Ping-pong Activation Banks (A and B)

**Context.** Each layer reads from one activation bank and writes to another.
Single-port BRAMs cannot read and write the same address simultaneously
without `READ_FIRST`/`WRITE_FIRST` modes.

**Decision.** Two activation BRAM banks, alternated:

```
load_in  → bank A
fc1: A → B
fc2: B → A
fc3: A → OSRAM
```

**Alternatives.**
- **Single bank with dual port**: 2× BRAM count.
- **Single bank with stall logic**: adds 1 cycle of stall per neuron.

**Consequences.**
- +Zero stalls during compute.
- +Each BRAM stays single-port (cheaper in resources).
- −fc3 cannot ping-pong (only one OSRAM), so a 4th layer would need
  another bank.

---

## ADR-006 — Bias SRAM as a Single Concatenated Array

**Context.** fc1 has 128 biases, fc2 has 64, fc3 has 10. We could give
each layer its own BSRAM.

**Decision.** Single BSRAM of depth `HIDDEN1 + HIDDEN2 + OUTPUT = 202`.
Each layer's compute pass adds `b_base_r` to compute its starting address.

**Alternatives.**
- 3 separate BSRAMs: more BRAM ports, more wiring.
- Per-output-neuron register file: huge area for 202 INT32 registers.

**Consequences.**
- +One BRAM, one read port, one write port.
- +Simplified DMA: load all 202 biases in one burst.
- −Caught a subtle bug where `BSRAM_AW = $clog2(128) = 7` truncated
  addresses 128-201 to 0-73. Fix: width sized from full depth (now
  `$clog2(202) = 8`). See `docs/Verification_Guide.md` §9 row #7.

---

## ADR-007 — DMA Loader as Behavioural Block (Simulation-Only)

**Context.** Final FPGA build needs an AXI4 master to fetch from DDR. But
for the prototype, an AXI4 implementation is ~3k LUTs and bring-up overhead.

**Decision.** `dma_loader.sv` exposes a simple `re/addr/rdata` interface
that the testbench drives directly via `dram_model.sv` (a `$readmemh`-loaded
BRAM array). The DMA interface is **architecturally equivalent** to AXI4
— just narrower in scope.

**Alternatives.**
- Full AXI4 from day one: blocks bring-up by weeks.
- No DMA at all (preload BRAMs in COE): less realistic, won't scale.

**Consequences.**
- +The accelerator is **fully verified end-to-end** without needing AXI IP.
- +Adding AXI later is mechanical: replace the testbench DRAM model with
  an AXI converter; the DUT's `dram_re/dram_addr/dram_rdata` ports stay
  the same.

---

## ADR-008 — Requant Multiplier as Compile-Time Constant

**Context.** Per-layer requant multipliers could be:
(a) hard-coded `\`define`s, (b) loaded from a separate `.mem` file at boot,
(c) written via CSR.

**Decision.** Compile-time `\`define`s in `nn_params_auto.svh`, regenerated
by `python/export_weights.py`.

**Alternatives.**
- Runtime-configurable (CSR or .mem): more flexible but adds 3 registers
  and a CSR interface.

**Consequences.**
- +Zero runtime overhead.
- +No extra BRAM or registers.
- −Retraining requires recompile. Acceptable: weights also change, so
  full re-synthesis is needed anyway.

---

## ADR-009 — Verification: Python as Golden Model

**Context.** Need a reference for self-checking that matches the RTL exactly.

**Decision.** `python/quantization.py:quant_layer()` is a 5-line function
that does exactly what the RTL PPU does (matmul → +bias → ·M_q → >>16 →
ReLU → INT8 saturate). The TB's `golden_output.mem` is produced by this
function.

**Alternatives.**
- C/C++ golden: harder to update, no good MNIST tooling.
- Hand-computed: error-prone, doesn't scale.
- PyTorch quantized inference: PyTorch's INT8 ops use different rounding,
  doesn't match RTL bit-exactly.

**Consequences.**
- +Tests are reproducible: same Python script → same golden every time.
- +When PyTorch trains a new model, the regression suite is fresh too.
- +Bugs in either side surface as 10/10 mismatches, easy to triage.

---

## ADR-010 — Reset Style: Asynchronous, Active-Low

**Context.** Xilinx best practice for 7-series is synchronous reset on
hot paths; UltraScale+ prefers async. We chose async.

**Decision.** All FFs use `always_ff @(posedge clk or negedge rst_n)`.

**Alternatives.**
- Synchronous reset only.
- No reset on data registers (only on control).

**Consequences.**
- +Compatible with both 7-series and UltraScale+.
- +Initialization is deterministic in simulation.
- −Slightly higher BRAM utilization on 7-series (the FDR primitive has
  sync reset).

---

## ADR-011 — Memory File Layout: 128-bit Packed Words

**Context.** PE array consumes 16 weights per cycle. WSRAM read port could
be 16 separate 8-bit ports, or one 128-bit port.

**Decision.** Single 128-bit port; 16 weights packed LSB-first into each word.

**Alternatives.**
- 16 separate single-byte SRAMs: more BRAM, more address generators.
- 64-bit port × 2-cycle reads: latency penalty.

**Consequences.**
- +Cleanest mapping to a single BRAM ×16 cascade in Vivado.
- +Matches DRAM model's `WSRAM_WORD_W = 128`.
- +`pack_mem.py` does the layout conversion (transparent to the user).

---

## ADR-012 — Test Inputs Pre-Committed to Git

**Context.** The 30 regression test cases need to be reproducible. Should
we commit `.mem` files or regenerate from Python?

**Decision.** Commit them. New contributors can regenerate via
`python/generate_test_vectors.py` but the canonical ones in
`examples/sample_run/` are static.

**Alternatives.**
- Always regenerate: blocks contributors who don't want to install PyTorch.
- Use Git LFS: overkill for 1.7 MB of text.

**Consequences.**
- +Repo works out of the box with zero setup.
- +Easy to diff if test vectors ever change.
- −Repo size +1.7 MB. Acceptable.

---

## ADR-013 — TCL for Vivado Automation, Bash/Python for Everything Else

**Context.** Vivado's only scripting interface is TCL. We could shell-out
from Bash, but TCL is the natural integration.

**Decision.** TCL scripts in `scripts/` for Vivado tasks. Bash in helper
loops. Python for everything pre-RTL.

**Consequences.**
- +Vivado batch-mode "just works" with `-source scripts/*.tcl`.
- +Bash regression script works in any POSIX shell.
- +Python is the lingua franca for ML and reference models.

---

## ADR-014 — License: MIT

**Context.** Choosing a permissive vs copyleft license affects adoption.

**Decision.** MIT.

**Alternatives.**
- BSD-2-Clause: equivalent.
- Apache-2.0: better patent protection, more bureaucratic.
- GPL: would block commercial use; not the goal.

**Consequences.**
- +Maximum adoption (companies can fork and modify).
- +Compatible with vendor IP that may live alongside this code.
- −No patent protection (we are not contributing patents anyway).
