# `examples/sample_run/`

A complete pre-built test suite + canonical sample inference results.

## Contents

```
sample_run/
├── README.md                            ← this file
└── test_inputs/                         ← 30 ready-to-use MNIST test cases
    ├── INDEX.md                         ← table of all 30 cases
    ├── REGRESSION_RESULTS.md            ← proof: 30/30 PASS
    ├── manifest.csv
    ├── use_test_case.sh / .tcl          ← one-shot loaders
    └── digit_<D>/idx_<I>/               ← per-case folders
        ├── input.mem  input_packed.mem
        ├── golden_output.mem
        ├── preview.png
        └── info.txt
```

## How to use

### One-shot from Bash

```bash
cd examples/sample_run/test_inputs
./use_test_case.sh 4 19         # load MNIST test idx=19 (digit 4)
```

### One-shot from Vivado TCL

```tcl
source examples/sample_run/test_inputs/use_test_case.tcl
use_test_case 7 0
```

### Manual

Copy the two `.mem` files from any
`digit_<D>/idx_<I>/` folder into your Vivado xsim working directory,
then **Run Simulation → Relaunch Simulation** (no recompile).

## Expected outcome

```
[TB] RTL    argmax (predicted digit) = <D>
[TB] Python argmax (predicted digit) = <D>
[TB] ***** PASS : 10/10 outputs match Python golden *****
```

For the "model gets it wrong" edge case (`digit_3/idx_0018` → "8"),
both Python and RTL agree on the wrong answer, which still PASSes the
bit-exact scoreboard.

## Regenerate

```bash
cd python
python generate_test_vectors.py --per_digit 3
```

This recreates all 30 cases (or any number per digit) from the trained
checkpoint.
