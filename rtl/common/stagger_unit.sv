// =============================================================================
// stagger_unit.sv -- input alignment / shift-reg delay for the PE array.
//
// Purpose
// -------
// The activation SRAM and the weight SRAM are READ ON THE SAME CYCLE, but the
// activation has to go through an extra mux (ping-pong bank select) and a
// broadcast fanout to N PEs, while the weight goes straight to the per-PE
// input register.  Without alignment, the activation arrives ONE cycle later
// than the weight at the PE input, corrupting every dot product.
//
// This module inserts a *parameterized* shift-register delay on the weight
// path so weight and activation are presented to the PE on the SAME cycle.
//
// It also gates `mac_en` and `mac_clr` with the same delay so the controller
// can fire those signals at "activation issue time" without thinking about
// the pipeline.
//
// DELAY is typically 1 (a single bank-mux cycle).  Make it 2 if you add an
// output register to the activation SRAM.
// =============================================================================

`timescale 1ns/1ps

module stagger_unit
    import nn_pkg::*;
#(
    parameter int DELAY  = 1,
    parameter int N      = ARRAY_SIZE,
    parameter int DATA_W = DATA_WIDTH
) (
    input  logic                          clk,
    input  logic                          rst_n,

    // raw (from SRAM read)
    input  logic signed [DATA_W-1:0]      a_raw,
    input  logic signed [N*DATA_W-1:0]    w_raw,
    input  logic                          en_raw,
    input  logic                          clr_raw,

    // aligned (to PE array)
    output logic signed [DATA_W-1:0]      a_out,
    output logic signed [N*DATA_W-1:0]    w_out,
    output logic                          en_out,
    output logic                          clr_out
);

    // Activation goes through (DELAY) registers, weight goes through 0 (already
    // delayed by the BRAM read pipeline).  We keep both as a parameterized
    // shift-reg so flipping DELAY rebalances the pipeline.
    generate
        if (DELAY == 0) begin : g_d0
            assign a_out   = a_raw;
            assign w_out   = w_raw;
            assign en_out  = en_raw;
            assign clr_out = clr_raw;
        end else begin : g_dn
            logic signed [DATA_W-1:0]      a_sr  [DELAY];
            logic signed [N*DATA_W-1:0]    w_sr  [DELAY];
            logic                          e_sr  [DELAY];
            logic                          c_sr  [DELAY];

            integer i;
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (i = 0; i < DELAY; i++) begin
                        a_sr[i] <= '0;
                        w_sr[i] <= '0;
                        e_sr[i] <= 1'b0;
                        c_sr[i] <= 1'b0;
                    end
                end else begin
                    a_sr[0] <= a_raw;
                    w_sr[0] <= w_raw;
                    e_sr[0] <= en_raw;
                    c_sr[0] <= clr_raw;
                    for (i = 1; i < DELAY; i++) begin
                        a_sr[i] <= a_sr[i-1];
                        w_sr[i] <= w_sr[i-1];
                        e_sr[i] <= e_sr[i-1];
                        c_sr[i] <= c_sr[i-1];
                    end
                end
            end

            assign a_out   = a_sr[DELAY-1];
            assign w_out   = w_sr[DELAY-1];
            assign en_out  = e_sr[DELAY-1];
            assign clr_out = c_sr[DELAY-1];
        end
    endgenerate

endmodule
