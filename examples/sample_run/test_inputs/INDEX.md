# Test Input Catalog
30 ready-to-use test cases covering all 10 MNIST digits (3 per digit).
Each folder contains a `input_packed.mem` + `golden_output.mem` you can
drop into Vivado's xsim working directory to test that case.

| Digit | Test idx | Folder | Python predicts | Correct? |
|-------|----------|--------|-----------------|----------|
| 0 | 3 | `digit_0/idx_0003` | 0 | OK |
| 0 | 10 | `digit_0/idx_0010` | 0 | OK |
| 0 | 13 | `digit_0/idx_0013` | 0 | OK |
| 1 | 2 | `digit_1/idx_0002` | 1 | OK |
| 1 | 5 | `digit_1/idx_0005` | 1 | OK |
| 1 | 14 | `digit_1/idx_0014` | 1 | OK |
| 2 | 1 | `digit_2/idx_0001` | 2 | OK |
| 2 | 35 | `digit_2/idx_0035` | 2 | OK |
| 2 | 38 | `digit_2/idx_0038` | 2 | OK |
| 3 | 18 | `digit_3/idx_0018` | 8 | model wrong |
| 3 | 30 | `digit_3/idx_0030` | 3 | OK |
| 3 | 32 | `digit_3/idx_0032` | 3 | OK |
| 4 | 4 | `digit_4/idx_0004` | 4 | OK |
| 4 | 6 | `digit_4/idx_0006` | 4 | OK |
| 4 | 19 | `digit_4/idx_0019` | 4 | OK |
| 5 | 8 | `digit_5/idx_0008` | 5 | OK |
| 5 | 15 | `digit_5/idx_0015` | 5 | OK |
| 5 | 23 | `digit_5/idx_0023` | 5 | OK |
| 6 | 11 | `digit_6/idx_0011` | 6 | OK |
| 6 | 21 | `digit_6/idx_0021` | 6 | OK |
| 6 | 22 | `digit_6/idx_0022` | 6 | OK |
| 7 | 0 | `digit_7/idx_0000` | 7 | OK |
| 7 | 17 | `digit_7/idx_0017` | 7 | OK |
| 7 | 26 | `digit_7/idx_0026` | 7 | OK |
| 8 | 61 | `digit_8/idx_0061` | 8 | OK |
| 8 | 84 | `digit_8/idx_0084` | 8 | OK |
| 8 | 110 | `digit_8/idx_0110` | 8 | OK |
| 9 | 7 | `digit_9/idx_0007` | 9 | OK |
| 9 | 9 | `digit_9/idx_0009` | 9 | OK |
| 9 | 12 | `digit_9/idx_0012` | 9 | OK |

## How to use any of these

1. Pick a folder (e.g. `digit_7/idx_0000/`)
2. Copy these two files into your Vivado project's xsim working dir:
   - `input_packed.mem`
   - `golden_output.mem`
3. Relaunch the simulation in Vivado (no recompile needed).
4. The TB will check the RTL output against `golden_output.mem` and print PASS/FAIL.
