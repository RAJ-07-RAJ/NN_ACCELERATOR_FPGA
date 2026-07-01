# `test_inputs/` — 30 Ready-to-Use MNIST Test Cases

Each subfolder is **one self-contained test case** consisting of an MNIST
test-set image, its packed RTL input, the Python golden output, a PNG
preview, and an info file.

You can swap any of these into the simulation **without recompiling**
the design — the testbench only re-reads `input_packed.mem` and
`golden_output.mem` on each simulation run.

---

## Folder layout

```
test_inputs/
├── INDEX.md                 ← table of all 30 cases (markdown)
├── manifest.csv             ← same data as CSV (for scripting)
├── use_test_case.sh         ← copy a case into Vivado's xsim dir (Bash)
├── use_test_case.tcl        ← same, but from inside the Vivado TCL console
├── README.md                ← this file
│
├── digit_0/                 ── 3 test cases per digit (0..9 = 30 total)
│   ├── idx_0003/
│   │   ├── input.mem           ← human-readable (784 lines of INT8 hex)
│   │   ├── input_packed.mem    ← what the TB reads (128-bit words)
│   │   ├── golden_output.mem   ← what the TB compares against
│   │   ├── preview.png         ← PNG of the actual image
│   │   └── info.txt            ← summary (true label, py prediction, logits)
│   ├── idx_0010/
│   └── idx_0013/
├── digit_1/
│   ├── idx_0002/
│   ├── idx_0005/
│   └── idx_0014/
... (digit_2 ... digit_9)
```

## Test case inventory

| Digit | MNIST indices         | Notes                                       |
|-------|-----------------------|---------------------------------------------|
| 0     | 3, 10, 13             | all correctly classified                    |
| 1     | 2, 5, 14              | all correctly classified                    |
| 2     | 1, 35, 38             | all correctly classified                    |
| 3     | 18, 30, 32            | **idx_0018 is misclassified as 8** (edge case)|
| 4     | 4, 6, 19              | all correctly classified                    |
| 5     | 8, 15, 23             | all correctly classified                    |
| 6     | 11, 21, 22            | all correctly classified                    |
| 7     | 0, 17, 26             | all correctly classified                    |
| 8     | 61, 84, 110           | all correctly classified                    |
| 9     | 7, 9, 12              | all correctly classified                    |

Full table in [`INDEX.md`](INDEX.md). Machine-readable in [`manifest.csv`](manifest.csv).

> 💡 **Edge case `digit_3/idx_0018`** is a "3" that the quantized model
> misclassifies as an "8". The RTL still produces the exact same wrong
> answer as Python — perfect for verifying that the RTL faithfully
> reproduces the Python golden even on hard inputs.

## How to load a test case

### Option 1 — Bash (Linux / macOS / WSL)

```bash
cd test_inputs
./use_test_case.sh 4 19
# -> copies digit_4/idx_0019/input_packed.mem + golden_output.mem into
#    ../vivado_proj/mnist_fpga_sim.sim/sim_1/behav/xsim/
```

You can pass a custom xsim path as the 3rd argument:

```bash
./use_test_case.sh 7 0 /my/project.sim/sim_1/behav/xsim
```

### Option 2 — From Vivado TCL console (any OS)

```tcl
source test_inputs/use_test_case.tcl
use_test_case 4 19
```

### Option 3 — Manual copy

Just drag-and-drop these two files from the test case folder into Vivado's
xsim working directory:

```
digit_4/idx_0019/input_packed.mem    →  <proj>.sim/sim_1/behav/xsim/
digit_4/idx_0019/golden_output.mem   →  <proj>.sim/sim_1/behav/xsim/
```

### After loading

In Vivado: **Run Simulation → Relaunch Simulation**. No recompile, no
re-elaboration. The TB will print a 10-row table comparing each output
neuron to the golden value, followed by:

```
[TB] RTL    argmax (predicted digit) = <D>
[TB] Python argmax (predicted digit) = <D>
[TB] ***** PASS : 10/10 outputs match Python golden *****
```

## Verifying you got the right output

Open `digit_<D>/idx_<I>/info.txt` for the case you loaded — it lists the
expected INT8 logits and the Python prediction. Compare against the
testbench output.

Or open `preview.png` to **see** the image you fed to the RTL.

## Want a different image?

If none of the 30 cases match what you want to test, generate a new one:

```bash
cd ../python              # parent project (sibling of mnist_fpga_vivado/)
python run_inference.py --idx <ANY_MNIST_INDEX>
```

That script writes a fresh `input_packed.mem` + `golden_output.mem` you
can drop straight into Vivado's xsim dir.
