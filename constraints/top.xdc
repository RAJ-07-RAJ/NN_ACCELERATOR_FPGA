# =============================================================================
# top.xdc -- timing constraints for nn_accel_top
#
# This file is board-agnostic: it constrains clock period and IO timing
# but does NOT assign physical pin locations (LOC). For a real board,
# create a board-specific XDC that adds LOCs and includes this file.
# =============================================================================

# ---- Primary clock --------------------------------------------------------
# Target: 100 MHz (10 ns period). The design closes timing on Artix-7 -1
# easily at this frequency. To explore higher Fmax, reduce -period.
create_clock -name clk -period 10.000 [get_ports clk]

# ---- Reset is async --------------------------------------------------------
set_false_path -from [get_ports rst_n]

# ---- CSR bus IO timing (treated as low-bandwidth, no tight constraint) ----
set_input_delay  -clock clk  2.0 [get_ports {reg_we reg_addr* reg_wdata*}]
set_output_delay -clock clk  2.0 [get_ports {reg_rdata* irq}]

# ---- DRAM IO (typically pipelined externally; allow 4 ns) -----------------
set_input_delay  -clock clk  4.0 [get_ports dram_rdata*]
set_output_delay -clock clk  4.0 [get_ports {dram_re dram_addr*}]

# ---- Output logits (debug port, no real-time requirement) ----------------
set_output_delay -clock clk  2.0 [get_ports {out_logits_packed* out_valid}]

# ---- Hint: keep weight SRAM as block RAM ---------------------------------
# Forces Vivado to use BRAM rather than distributed RAM for the WSRAM,
# regardless of how the inference engine sizes it.
set_property RAM_STYLE block [get_cells -hier -filter {NAME =~ *u_wsram*}]

# ---- (Optional) Set IO standards for FPGA bring-up -----------------------
# Uncomment and customize for your board:
# set_property IOSTANDARD LVCMOS33 [get_ports clk]
# set_property IOSTANDARD LVCMOS33 [get_ports rst_n]
# set_property PACKAGE_PIN E3 [get_ports clk]    # Arty A7 100 MHz pin

# ---- DRC waivers (none currently) -----------------------------------------
