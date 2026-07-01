#!/usr/bin/env bash
# =============================================================================
# use_test_case.sh -- copy a chosen test case into Vivado's xsim working dir
#
# Usage:
#     ./use_test_case.sh <digit> <idx> [<xsim_dir>]
#
# Examples:
#     # copy MNIST test idx=42 (a 4) into the default Vivado proj dir
#     ./use_test_case.sh 4 19
#
#     # specify a custom xsim path
#     ./use_test_case.sh 7 0 /path/to/vivado_proj/proj.sim/sim_1/behav/xsim
#
# What it does:
#   1. Finds test_inputs/digit_<D>/idx_<IDX>/ in this folder
#   2. Copies input_packed.mem + golden_output.mem into <xsim_dir>
#   3. Reminds you that weights_packed.mem + bias_packed.mem must also be there
# =============================================================================
set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <digit 0..9> <mnist_idx> [<xsim_dir>]"
    exit 1
fi

DIGIT="$1"
IDX="$(printf '%04d' "$2")"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Default xsim path = $PROJ_ROOT/vivado_proj/mnist_fpga_sim.sim/sim_1/behav/xsim
XSIM_DIR="${3:-$PROJ_ROOT/vivado_proj/mnist_fpga_sim.sim/sim_1/behav/xsim}"

CASE_DIR="$SCRIPT_DIR/digit_$DIGIT/idx_$IDX"
if [ ! -d "$CASE_DIR" ]; then
    echo "ERROR: no such test case: $CASE_DIR"
    echo ""
    echo "Available test cases:"
    ls -d "$SCRIPT_DIR"/digit_*/idx_* 2>/dev/null | sed "s|$SCRIPT_DIR/||"
    exit 2
fi

mkdir -p "$XSIM_DIR"
cp -v "$CASE_DIR/input_packed.mem"  "$XSIM_DIR/"
cp -v "$CASE_DIR/golden_output.mem" "$XSIM_DIR/"

# also copy weights/bias if missing
for f in weights_packed.mem bias_packed.mem; do
    if [ ! -f "$XSIM_DIR/$f" ]; then
        cp -v "$PROJ_ROOT/mem/$f" "$XSIM_DIR/"
    fi
done

echo ""
echo "================================================================"
cat "$CASE_DIR/info.txt"
echo "================================================================"
echo ""
echo "Done. In Vivado: Run Simulation -> Relaunch Simulation (no recompile)."
