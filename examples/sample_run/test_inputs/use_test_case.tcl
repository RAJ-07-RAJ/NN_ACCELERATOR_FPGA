# =============================================================================
# use_test_case.tcl -- copy a chosen test case into Vivado's xsim working dir
#                      (pure TCL version of use_test_case.sh)
#
# Usage (from Vivado TCL console with the project open):
#     source <path>/test_inputs/use_test_case.tcl
#     use_test_case <digit> <idx>
#
# Examples:
#     use_test_case 7 0          ;# MNIST test idx 0 (a "7")
#     use_test_case 4 19         ;# MNIST test idx 19 (a "4")
#     use_test_case 3 18         ;# Edge case -- model misclassifies this 3 as 8
# =============================================================================

proc use_test_case {digit idx} {
    set script_path [file dirname [file normalize [info script]]]
    set proj_root   [file dirname $script_path]

    # find the matching test case folder
    set idx_str [format "%04d" $idx]
    set case_dir [file join $script_path "digit_$digit" "idx_$idx_str"]
    if {![file isdirectory $case_dir]} {
        puts "ERROR: no test case at $case_dir"
        return
    }

    # locate Vivado's xsim run dir for the current project
    set sim_dir [get_property DIRECTORY [current_project]]
    set proj_name [get_property NAME [current_project]]
    set xsim_dir [file join $sim_dir "$proj_name.sim" "sim_1" "behav" "xsim"]
    file mkdir $xsim_dir

    # copy the case files + ensure weights/bias are present
    foreach f {input_packed.mem golden_output.mem} {
        file copy -force [file join $case_dir $f] $xsim_dir
    }
    foreach f {weights_packed.mem bias_packed.mem} {
        set dst [file join $xsim_dir $f]
        if {![file exists $dst]} {
            file copy -force [file join $proj_root "mem" $f] $dst
        }
    }

    puts ""
    puts "============================================================"
    puts "Loaded test case: digit $digit, MNIST idx $idx"
    puts [exec cat [file join $case_dir "info.txt"]]
    puts "============================================================"
    puts "Now: Run Simulation -> Relaunch Simulation (no recompile)."
}

puts "Defined proc: use_test_case <digit> <idx>"
puts "Example: use_test_case 7 0"
