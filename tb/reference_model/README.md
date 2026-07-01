# `tb/reference_model/`

The behavioural models used **only** during simulation.

## Files

| File             | Purpose                                                   |
|------------------|-----------------------------------------------------------|
| `dram_model.sv`  | Behavioural DRAM. Loads `weights/bias/input_packed.mem` at elaboration via `$readmemh`. Exposes a 1-cycle read port. |

## How it connects

```
        ┌─────────────────────────────────┐
        │     tb_nn_accel_top.sv          │
        │                                  │
        │   ┌─────────────────┐           │
        │   │  dram_model     │           │
        │   │  (this folder)  │           │
        │   └────────┬────────┘           │
        │            │ dram_re / addr     │
        │            ▼                    │
        │   ┌─────────────────┐           │
        │   │ DUT (nn_accel_top)│         │
        │   └─────────────────┘           │
        └──────────────────────────────────┘
```

## Python golden reference

The **bit-exact Python reference** is implemented in:

- `python/golden_model.py` — `GoldenModel` class
- `python/quantization.py` — `quant_layer()` function

Together these produce the `golden_output.mem` file that the testbench
compares the DUT's OSRAM contents against.
