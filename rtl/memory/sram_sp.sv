// =============================================================================
// sram_sp.sv -- generic single-port synchronous SRAM, inferable as BRAM.
//
// Synthesis notes:
// - One write port, one read port, both registered (standard BRAM template).
// - Read-during-write returns OLD data ("read-first") -- safest for FSMs that
//   would otherwise see X on the read path during the write cycle.
// - Optional INIT_FILE loads $readmemh at elaboration; useful for sim and
//   for FPGA bring-up where weights live in BRAM init.
// =============================================================================

`timescale 1ns/1ps

module sram_sp #(
    parameter int DATA_W    = 8,
    parameter int DEPTH     = 1024,
    parameter int ADDR_W    = (DEPTH<=1) ? 1 : $clog2(DEPTH),
    parameter string INIT_FILE = ""
) (
    input  logic              clk,
    input  logic              en,
    input  logic              we,
    input  logic [ADDR_W-1:0] addr,
    input  logic [DATA_W-1:0] din,
    output logic [DATA_W-1:0] dout
);

    // Memory array — synthesis tools will infer BRAM for DEPTH >= ~512.
    (* ram_style = "block" *) logic [DATA_W-1:0] mem [0:DEPTH-1];

    initial begin
        if (INIT_FILE != "") begin
            $display("[sram_sp] $readmemh from %s", INIT_FILE);
            $readmemh(INIT_FILE, mem);
        end
    end

    always_ff @(posedge clk) begin
        if (en) begin
            if (we) mem[addr] <= din;
            dout <= mem[addr];   // read-first
        end
    end

endmodule
