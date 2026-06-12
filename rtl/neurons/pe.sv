// =============================================================================
// pe.sv -- single Processing Element (signed INT8 x INT8 MAC).
//
// Simplified, single-cycle MAC:
//     when clr=1 (sync):     acc <= 0
//     else when en=1:        acc <= acc + a_in * b_in       (signed)
//     else:                  acc <= acc                     (hold)
//
// One register stage (acc).  The multiplier is combinational; for INT8 x INT8
// this comfortably closes 200 MHz on a 7-series FPGA (DSP48 packs it into a
// single tile with mul + add).  Removing the 2-stage pipeline keeps the
// control timing trivial: the FSM issues clr at the start of a tile, then
// en for IN cycles, and the final acc is valid exactly 1 cycle after the
// last en pulse.
// =============================================================================

`timescale 1ns/1ps

module pe #(
    parameter int DATA_W = 8,
    parameter int ACC_W  = 32
) (
    input  logic                       clk,
    input  logic                       rst_n,

    input  logic                       en,        // accumulate this cycle
    input  logic                       clr,       // synchronous clear (priority over en)
    input  logic signed [DATA_W-1:0]   a_in,
    input  logic signed [DATA_W-1:0]   b_in,

    output logic signed [ACC_W-1:0]    acc
);

    logic signed [ACC_W-1:0] acc_q;
    logic signed [2*DATA_W-1:0] prod;

    // signed * signed -> signed (Verilog evaluates this correctly when both
    // operands are declared `signed`).
    assign prod = a_in * b_in;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_q <= '0;
        end else if (clr) begin
            // clr takes priority; if en is also high this cycle we still
            // start fresh (data on this cycle is ignored, controller will
            // issue the first MAC on the next cycle).
            acc_q <= '0;
        end else if (en) begin
            // sign-extend the 16-bit product into the 32-bit accumulator
            acc_q <= acc_q + {{(ACC_W-2*DATA_W){prod[2*DATA_W-1]}}, prod};
        end
    end

    assign acc = acc_q;

endmodule
