// =============================================================================
// output_monitor.sv -- snoops OSRAM contents and publishes events
//
// Captures the 10 INT8 output bytes when `done` asserts, and emits a
// "transaction" struct to the scoreboard.
//
// Used by tb_nn_accel_top.sv via instantiation; for procedural TBs the
// scoreboard reads `out_logits_packed` directly.
// =============================================================================
`timescale 1ns/1ps

module output_monitor
    import nn_pkg::*;
(
    input  logic                                clk,
    input  logic                                rst_n,
    input  logic                                done_pulse,
    input  logic [OUTPUT_SIZE*DATA_WIDTH-1:0]   out_logits_packed,

    // Outgoing transaction
    output logic                                txn_valid,
    output logic signed [DATA_WIDTH-1:0]        txn_logits [0:OUTPUT_SIZE-1],
    output logic [3:0]                          txn_argmax
);

    logic [3:0] best_idx;
    logic signed [DATA_WIDTH-1:0] best_val;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            txn_valid  <= 1'b0;
            txn_argmax <= '0;
            for (int i = 0; i < OUTPUT_SIZE; i++) txn_logits[i] <= '0;
        end else begin
            txn_valid <= done_pulse;
            if (done_pulse) begin
                best_idx = 0;
                best_val = $signed(out_logits_packed[0 +: DATA_WIDTH]);
                for (int i = 0; i < OUTPUT_SIZE; i++) begin
                    txn_logits[i] = $signed(out_logits_packed[i*DATA_WIDTH +: DATA_WIDTH]);
                    if (txn_logits[i] > best_val) begin
                        best_val = txn_logits[i];
                        best_idx = i;
                    end
                end
                txn_argmax <= best_idx;
                $display("[mon ] t=%0t  argmax=%0d  logits=%p",
                         $time, best_idx, txn_logits);
            end
        end
    end

endmodule
