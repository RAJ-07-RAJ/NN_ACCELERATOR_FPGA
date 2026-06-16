// =============================================================================
// ppu.sv -- Post Processing Unit: bias add + requantize + ReLU + saturate.
//
// Per output neuron, given a 32-bit accumulator `acc`:
//     y = acc + bias                          (INT32)
//     y = (y * M_q) >>> FRACTION_BITS         (signed arithmetic shift)
//     if relu_en && y<0 : y = 0
//     y = clip(y, -128, 127)                  (INT8 saturation)
//
// Pipelining (3 cycles, fully registered, one DSP for the multiply):
//     S0 : add bias
//     S1 : multiply by M_q (signed 33b * 32b -> 65b, but DSP48 handles 25x18
//          unsigned/signed easily for this range; we narrow upstream)
//     S2 : arithmetic shift + ReLU + saturate
//
// The PPU processes ONE neuron per cycle, fed by the drain unit.
// =============================================================================

`timescale 1ns/1ps

module ppu
    import nn_pkg::*;
#(
    parameter int ACC_W   = ACC_WIDTH,
    parameter int BIAS_W  = BIAS_WIDTH,
    parameter int REQ_W   = REQ_WIDTH,
    parameter int DATA_W  = DATA_WIDTH,
    parameter int FRAC_B  = FRACTION_BITS
) (
    input  logic                       clk,
    input  logic                       rst_n,

    input  logic                       in_valid,
    input  logic signed [ACC_W-1:0]    in_acc,
    input  logic signed [BIAS_W-1:0]   in_bias,
    input  logic signed [REQ_W-1:0]    m_q,
    input  logic                       relu_en,

    output logic                       out_valid,
    output logic signed [DATA_W-1:0]   out_data
);

    // ---- S0 : bias add ----
    localparam int SUM_W = (ACC_W > BIAS_W ? ACC_W : BIAS_W) + 1;
    logic signed [SUM_W-1:0] s0_sum;
    logic                    s0_v;
    logic signed [REQ_W-1:0] s0_m;
    logic                    s0_relu;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0_sum  <= '0;
            s0_v    <= 1'b0;
            s0_m    <= '0;
            s0_relu <= 1'b0;
        end else begin
            s0_sum  <= $signed(in_acc) + $signed(in_bias);
            s0_v    <= in_valid;
            s0_m    <= m_q;
            s0_relu <= relu_en;
        end
    end

    // ---- S1 : multiply by requant multiplier ----
    // Worst-case widths: SUM_W (~33) * REQ_W (32) -> ~65 bits signed.
    localparam int MUL_W = SUM_W + REQ_W;
    logic signed [MUL_W-1:0] s1_mul;
    logic                    s1_v;
    logic                    s1_relu;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_mul  <= '0;
            s1_v    <= 1'b0;
            s1_relu <= 1'b0;
        end else begin
            s1_mul  <= s0_sum * s0_m;
            s1_v    <= s0_v;
            s1_relu <= s0_relu;
        end
    end

    // ---- S2 : arithmetic shift + ReLU + saturate ----
    logic signed [MUL_W-1:0] shifted;
    logic signed [DATA_W-1:0] sat;
    logic                     v2;

    always_comb begin
        // signed arithmetic shift right by FRAC_B
        shifted = s1_mul >>> FRAC_B;
        // ReLU
        if (s1_relu && shifted < 0)
            shifted = '0;
        // saturate to INT8
        if (shifted >  127)            sat = 8'sd127;
        else if (shifted < -128)       sat = -8'sd128;
        else                           sat = shifted[DATA_W-1:0];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_data  <= '0;
            out_valid <= 1'b0;
            v2        <= 1'b0;
        end else begin
            out_data  <= sat;
            out_valid <= s1_v;
            v2        <= s1_v;
        end
    end

endmodule
