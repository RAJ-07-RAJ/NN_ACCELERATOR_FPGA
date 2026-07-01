// =============================================================================
// scoreboard.sv -- compares output monitor txns against the golden reference
//
// In this project the scoreboard is also implemented inline in
// tb_nn_accel_top.sv (procedural style).  This module wraps the same
// logic as a reusable block for future UVM-style integration.
// =============================================================================
`timescale 1ns/1ps

module scoreboard
    import nn_pkg::*;
#(
    parameter string GOLDEN_FILE = "golden_output.mem"
) (
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          txn_valid,
    input  logic signed [DATA_WIDTH-1:0]  txn_logits [0:OUTPUT_SIZE-1],
    output logic                          all_pass
);

    logic [DATA_WIDTH-1:0]         golden_raw [0:OUTPUT_SIZE-1];
    logic signed [DATA_WIDTH-1:0]  golden     [0:OUTPUT_SIZE-1];

    initial begin
        $readmemh(GOLDEN_FILE, golden_raw);
        for (int i = 0; i < OUTPUT_SIZE; i++)
            golden[i] = golden_raw[i];
    end

    int mismatches;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mismatches <= 0;
            all_pass   <= 1'b0;
        end else if (txn_valid) begin
            mismatches = 0;
            $display("=================================================");
            $display("[scbd] RTL vs GOLDEN");
            $display("=================================================");
            for (int i = 0; i < OUTPUT_SIZE; i++) begin
                if (txn_logits[i] !== golden[i]) begin
                    $display("[scbd]  [%0d] got=%0d  exp=%0d  MISMATCH",
                             i, txn_logits[i], golden[i]);
                    mismatches++;
                end else begin
                    $display("[scbd]  [%0d] got=%0d  exp=%0d  OK", i, txn_logits[i], golden[i]);
                end
            end
            all_pass <= (mismatches == 0);
            if (mismatches == 0)
                $display("[scbd] *** PASS : 10/10 outputs match Python golden ***");
            else
                $display("[scbd] *** FAIL : %0d mismatches ***", mismatches);
        end
    end

endmodule
