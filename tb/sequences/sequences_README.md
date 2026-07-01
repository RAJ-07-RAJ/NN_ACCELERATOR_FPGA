# `tb/sequences/` — Test Sequences

This folder holds **reusable stimulus sequences** that can be invoked from
the top testbench.

## Current sequences

| File / class               | Purpose                                          |
|----------------------------|--------------------------------------------------|
| (inline in `tb_nn_accel_top.sv`) | Default sequence: reset → program CSR → start → poll done |

## Planned sequences (v1.3+)

| Sequence                      | Purpose                                          |
|-------------------------------|--------------------------------------------------|
| `back_to_back_inferences.sv`  | Issue N inferences without reset between          |
| `soft_reset_mid_run.sv`       | Soft-reset during compute, verify recovery        |
| `bad_input_ptr.sv`            | Negative test: invalid INPUT_PTR                  |
| `irq_handshake.sv`            | Verify done IRQ assertion + W1C clear sequence   |
| `random_csr_sequencer.sv`     | Constrained-random CSR poking                     |

## How to add a new sequence

1. Create `<your_sequence>.sv` in this folder.
2. Use the `csr_driver_pkg::csr_write/read/poll_until_done` tasks
   from `tb/drivers/csr_driver.sv`.
3. Reference your sequence from a new top TB in `tb/top_tb/`.

Example template:

```systemverilog
// tb/sequences/my_sequence.sv
import csr_driver_pkg::*;

task automatic my_sequence(virtual csr_drv_if vif, input int seed);
    logic [31:0] rd;
    $display("[seq:my_sequence] starting (seed=%0d)", seed);

    csr_write(vif, 8'h08, 32'h0002_0000);  // INPUT_PTR
    csr_write(vif, 8'h00, 32'h0000_0001);  // start
    poll_until_done(vif);
    csr_read(vif, 8'h1C, rd);
    $display("[seq:my_sequence] cycles = %0d", rd);
endtask
```
