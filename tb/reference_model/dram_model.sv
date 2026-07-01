// =============================================================================
// dram_model.sv -- simulation-only "DRAM" backing store for the DMA.
//
// Layout (matches main_fsm.sv programming):
//     0x0000_0000 .. 0x0000_FFFF   packed weights (WSRAM_WORD_W bits per word)
//     0x0001_0000 .. 0x0001_FFFF   biases (one INT32 per word, zero-padded)
//     0x0002_0000 .. 0x0002_FFFF   one INT8 input image (zero-padded)
//
// We load each section from a separate .mem file produced by quantize.py.
// =============================================================================

`timescale 1ns/1ps

module dram_model
    import nn_pkg::*;
#(
    parameter string W_FILE = "weights_packed.mem",
    parameter string B_FILE = "bias_packed.mem",
    parameter string I_FILE = "input_packed.mem",
    parameter int    DEPTH  = 1 << 18
) (
    input  logic                       clk,
    input  logic                       re,
    input  logic [31:0]                addr,
    output logic [WSRAM_WORD_W-1:0]    rdata
);

    logic [WSRAM_WORD_W-1:0] mem [0:DEPTH-1];

    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
        if (W_FILE != "") begin $display("[dram] load %s @ 0x00000", W_FILE); $readmemh(W_FILE, mem, 32'h00000); end
        if (B_FILE != "") begin $display("[dram] load %s @ 0x10000", B_FILE); $readmemh(B_FILE, mem, 32'h10000); end
        if (I_FILE != "") begin $display("[dram] load %s @ 0x20000", I_FILE); $readmemh(I_FILE, mem, 32'h20000); end
    end

    always_ff @(posedge clk) begin
        if (re) rdata <= mem[addr[$clog2(DEPTH)-1:0]];
    end

endmodule
