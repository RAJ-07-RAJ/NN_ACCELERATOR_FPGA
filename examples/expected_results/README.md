# `examples/expected_results/`

Canonical "PASS" output of the testbench, for visual comparison.

## Single-image expected console output

```
[dram] load weights_packed.mem @ 0x00000
[dram] load bias_packed.mem    @ 0x10000
[dram] load input_packed.mem   @ 0x20000
[TB] t=165000 : starting inference (writing CTRL.start)
[TB] t=149905000 : done detected after 7486 poll iterations

=================================================
[TB]        RTL OUTPUT vs PYTHON GOLDEN
=================================================
[TB]  idx |   RTL (hex /  dec) | GOLDEN (hex /  dec) | status
[TB] -----+--------------------+---------------------+--------
[TB]   0  |     f8  /   -8    |     f8  /   -8     |       OK
[TB]   1  |     fc  /   -4    |     fc  /   -4     |       OK
[TB]   2  |     08  /    8    |     08  /    8     |       OK
[TB]   3  |     04  /    4    |     04  /    4     |       OK
[TB]   4  |     e8  /  -24   |     e8  /  -24    |       OK
[TB]   5  |     f8  /   -8    |     f8  /   -8     |       OK
[TB]   6  |     e5  /  -27   |     e5  /  -27    |       OK
[TB]   7  |     14  /   20   |     14  /   20    |       OK
[TB]   8  |     f7  /   -9    |     f7  /   -9     |       OK
[TB]   9  |     fe  /   -2    |     fe  /   -2     |       OK

[TB] RTL    argmax (predicted digit) = 7
[TB] Python argmax (predicted digit) = 7
[TB] cycle_count register = 14969 cycles  (~149 us @ 100 MHz)

[TB] ***** PASS : 10/10 outputs match Python golden *****
```

## Regression expected output (all 30 cases)

```
digit_0/idx_0003          PASS
digit_0/idx_0010          PASS
digit_0/idx_0013          PASS
digit_1/idx_0002          PASS
...
digit_9/idx_0012          PASS

==========================================
 REGRESSION SUMMARY : 30/30 PASS
==========================================
```

## If your output differs

See [`docs/Debug_Guide.md`](../../docs/Debug_Guide.md) for the
decision tree to diagnose mismatches.
