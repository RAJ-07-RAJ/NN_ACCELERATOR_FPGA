// =============================================================================
// dma_loader.sv -- simple DRAM->on-chip-SRAM burst loader.
//
// This is a *behavioural* DMA suitable for simulation and for FPGAs where the
// "DRAM" is actually preloaded into a BRAM via $readmemh.  In a real product
// it would be replaced by an AXI4 master that issues burst reads.
//
// Programming interface (from the main FSM):
//     start     : 1-cycle pulse
//     src_addr  : starting word address in the source (DRAM model)
//     dst_addr  : starting word address in the destination SRAM
//     length    : number of words to transfer
//     dst_sel   : 0=Weight SRAM, 1=Activation SRAM bank A, 2=bank B, 3=Bias
//
// One word per cycle.  done is held for one cycle when length words have been
// written.
// =============================================================================

`timescale 1ns/1ps

module dma_loader
    import nn_pkg::*;
#(
    parameter int WORD_W = WSRAM_WORD_W
) (
    input  logic                       clk,
    input  logic                       rst_n,

    // CSR side
    input  logic                       start,
    input  logic [31:0]                src_addr,
    input  logic [31:0]                dst_addr,
    input  logic [31:0]                length,
    input  logic [1:0]                 dst_sel,
    output logic                       busy,
    output logic                       done,

    // "DRAM" read port (single-cycle SRAM model in sim)
    output logic [31:0]                dram_addr,
    output logic                       dram_re,
    input  logic [WORD_W-1:0]          dram_rdata,    // widest port; narrower
                                                      // dests just take the LSBs

    // destination write ports
    output logic                       w_we, a_we, b_we, o_we,
    output logic [31:0]                w_addr, a_addr, b_addr, o_addr,
    output logic [WORD_W-1:0]          w_din,
    output logic [DATA_WIDTH-1:0]      a_din,
    output logic [BIAS_WIDTH-1:0]      b_din,
    output logic [DATA_WIDTH-1:0]      o_din,
    output logic                       a_bank_sel    // 0=A, 1=B (for activation)
);

    typedef enum logic [1:0] {S_IDLE, S_RUN, S_DRAIN, S_DONE} state_t;
    state_t state, nstate;

    logic [31:0] cnt;          // words written
    logic [31:0] rd_cnt;       // reads issued
    logic        rd_valid_d1;  // dram_rdata is valid one cycle after rd

    // ---- read address generator ----
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            cnt         <= '0;
            rd_cnt      <= '0;
            rd_valid_d1 <= 1'b0;
        end else begin
            state <= nstate;
            case (state)
                S_IDLE: begin
                    cnt    <= '0;
                    rd_cnt <= '0;
                    rd_valid_d1 <= 1'b0;
                end
                S_RUN: begin
                    if (rd_cnt < length) rd_cnt <= rd_cnt + 32'd1;
                    rd_valid_d1 <= (rd_cnt < length); // we issued a read this cycle
                    if (rd_valid_d1) cnt <= cnt + 32'd1; // we WRITE next cycle
                end
                S_DRAIN: begin
                    rd_valid_d1 <= 1'b0;
                    if (rd_valid_d1) cnt <= cnt + 32'd1;
                end
                default: ;
            endcase
        end
    end

    always_comb begin
        nstate = state;
        unique case (state)
            S_IDLE : if (start)                   nstate = S_RUN;
            S_RUN  : if (rd_cnt == length)        nstate = S_DRAIN;
            S_DRAIN: if (cnt    == length)        nstate = S_DONE;
            S_DONE :                              nstate = S_IDLE;
            default:                              nstate = S_IDLE;
        endcase
    end

    // ---- DRAM read port ----
    assign dram_re   = (state == S_RUN) && (rd_cnt < length);
    assign dram_addr = src_addr + rd_cnt;

    // ---- destination write fanout ----
    // Data arrives one cycle after the read was issued (BRAM latency).
    logic [31:0] wr_addr;
    assign wr_addr = dst_addr + cnt;

    always_comb begin
        w_we = 1'b0; a_we = 1'b0; b_we = 1'b0; o_we = 1'b0;
        w_addr = '0; a_addr = '0; b_addr = '0; o_addr = '0;
        w_din  = '0; a_din  = '0; b_din  = '0; o_din  = '0;
        a_bank_sel = dst_sel[0]; // re-use LSB for which bank

        if (rd_valid_d1 && (state == S_RUN || state == S_DRAIN)) begin
            unique case (dst_sel)
                2'd0: begin   // Weight SRAM (packed word)
                    w_we   = 1'b1;
                    w_addr = wr_addr;
                    w_din  = dram_rdata;
                end
                2'd1, 2'd2: begin   // Activation SRAM bank A or B
                    a_we   = 1'b1;
                    a_addr = wr_addr;
                    a_din  = dram_rdata[DATA_WIDTH-1:0];
                    a_bank_sel = (dst_sel == 2'd2);
                end
                2'd3: begin   // Bias SRAM (INT32)
                    b_we   = 1'b1;
                    b_addr = wr_addr;
                    b_din  = dram_rdata[BIAS_WIDTH-1:0];
                end
                default: ;
            endcase
        end
    end

    assign busy = (state != S_IDLE) && (state != S_DONE);
    assign done = (state == S_DONE);

endmodule
