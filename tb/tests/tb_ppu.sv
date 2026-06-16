// =============================================================================
// tb_ppu.sv -- check PPU math against the same closed-form quantize_layer used
// in python/quantize.py.
// =============================================================================

`timescale 1ns/1ps

module tb_ppu;
    import nn_pkg::*;

    logic clk = 0; always #5 clk = ~clk;
    logic rst_n = 0;

    logic                          v_in, v_out;
    logic signed [ACC_WIDTH-1:0]   acc_in;
    logic signed [BIAS_WIDTH-1:0]  bias_in;
    logic signed [REQ_WIDTH-1:0]   m_q;
    logic                          relu_en;
    logic signed [DATA_WIDTH-1:0]  data_out;

    ppu dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(v_in), .in_acc(acc_in), .in_bias(bias_in),
        .m_q(m_q), .relu_en(relu_en),
        .out_valid(v_out), .out_data(data_out)
    );

    // golden -- matches RTL exactly
    function automatic int golden_q(input longint acc, input longint bias,
                                    input longint M, input int frac,
                                    input bit relu);
        longint s = acc + bias;
        longint y = (s * M) >>> frac;
        if (relu && y < 0) y = 0;
        if (y > 127)  y = 127;
        if (y < -128) y = -128;
        return int'(y);
    endfunction

    int errors = 0;
    initial begin
        $dumpfile("tb_ppu.vcd"); $dumpvars(0, tb_ppu);
        v_in=0; acc_in=0; bias_in=0; m_q=0; relu_en=0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (3) @(posedge clk);

        for (int t = 0; t < 200; t++) begin
            int a = $urandom_range(0, 65535) - 32768;
            int b = $urandom_range(0, 1000)  - 500;
            int M = $urandom_range(1, 65535);
            int r = ($urandom & 1);
            int expected;

            @(negedge clk);
            v_in = 1; acc_in = a; bias_in = b; m_q = M; relu_en = r;
            @(negedge clk); v_in = 0;
            repeat (4) @(posedge clk); // wait 3-stage pipe

            expected = golden_q(a, b, M, FRACTION_BITS, r);
            if (data_out !== expected[DATA_WIDTH-1:0]) begin
                $display("[tb_ppu] FAIL  acc=%0d bias=%0d M=%0d relu=%0b  got=%0d exp=%0d",
                         a, b, M, r, data_out, expected);
                errors++;
            end
        end

        if (errors == 0) $display("[tb_ppu] *** PASS *** (200 random)");
        else             $display("[tb_ppu] *** FAIL : %0d errors ***", errors);
        $finish;
    end
endmodule
