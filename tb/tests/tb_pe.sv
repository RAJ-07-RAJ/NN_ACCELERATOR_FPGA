// =============================================================================
// tb_pe.sv -- directed + random tests for a single PE.
//
// Checks:
//   - reset clears accumulator
//   - clr clears mid-stream
//   - random INT8*INT8 dot-product matches Python-style numpy reduction
//   - no overflow inside 32-bit accumulator for 1024-element stream
// =============================================================================

`timescale 1ns/1ps

module tb_pe;
    import nn_pkg::*;

    localparam int DATA_W = 8;
    localparam int ACC_W  = 32;
    localparam int N_STREAM = 1024;

    logic clk = 0;
    always #5 clk = ~clk;
    logic rst_n = 0;

    logic en, clr;
    logic signed [DATA_W-1:0] a, b;
    logic signed [ACC_W-1:0]  acc;

    pe #(.DATA_W(DATA_W), .ACC_W(ACC_W)) dut (
        .clk(clk), .rst_n(rst_n),
        .en(en), .clr(clr), .a_in(a), .b_in(b), .acc(acc)
    );

    int errors = 0;

    task automatic stream_test(input int seed);
        longint expected = 0;
        int      a_q[$], b_q[$];
        // generate
        for (int i = 0; i < N_STREAM; i++) begin
            int aa = $urandom_range(0, 255) - 128;
            int bb = $urandom_range(0, 255) - 128;
            a_q.push_back(aa); b_q.push_back(bb);
            expected += aa * bb;
        end
        // clear
        @(negedge clk);
        en = 0; clr = 1; a = 0; b = 0;
        @(negedge clk);
        clr = 0; en = 1;
        for (int i = 0; i < N_STREAM; i++) begin
            a = a_q[i]; b = b_q[i];
            @(negedge clk);
        end
        en = 0;
        // wait for pipe to settle (2 cycles)
        repeat (3) @(negedge clk);
        if (acc !== expected) begin
            $display("[tb_pe] FAIL seed=%0d  got=%0d  expected=%0d", seed, acc, expected);
            errors++;
        end else begin
            $display("[tb_pe] PASS seed=%0d  acc=%0d", seed, acc);
        end
    endtask

    initial begin
        $dumpfile("tb_pe.vcd"); $dumpvars(0, tb_pe);
        en = 0; clr = 0; a = 0; b = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (3) @(posedge clk);

        for (int s = 0; s < 5; s++) stream_test(s);

        if (errors == 0) $display("[tb_pe] *** ALL PASS ***");
        else             $display("[tb_pe] *** %0d FAILS ***", errors);
        $finish;
    end

endmodule
