// =============================================================================
// pe_array.sv -- N parallel PEs sharing the same activation, fed N weights.
//
// Dataflow: WEIGHT STATIONARY (per output-tile).
//
//   For one output tile of N=ARRAY_SIZE neurons, we stream the activation
//   vector through the array one element per cycle:
//       cycle t : broadcast a[t] to all N PEs,
//                 give PE_k its own weight  W[tile*N + k, t]
//                 every PE does acc_k += a[t] * W_k
//   After IN_SIZE cycles all N accumulators hold the dot products for that
//   tile.  We then move to the next tile.
//
// Why weight-stationary?
//   - The MNIST FC layers are SMALL and have a long input dimension (784),
//     so loading weights is the dominant cost.  Keeping weights resident in
//     PE-side registers (well, in the per-cycle weight word delivered from
//     packed WSRAM) and STREAMING activations minimises weight bandwidth.
//   - Output-stationary would also work but needs an N-deep accumulator
//     scratchpad and a separate reduction tree.
//   - Row-stationary (Eyeriss) wins on big CONV layers where filter reuse
//     across spatial positions is huge; for FC it brings no benefit.
//
// Latency:
//   compute_cycles_per_tile = IN_SIZE + PE_PIPE_DEPTH (=2)
//   total per layer         = ceil(OUT_SIZE/N) * compute_cycles_per_tile
// =============================================================================

`timescale 1ns/1ps

module pe_array
    import nn_pkg::*;
#(
    parameter int N      = ARRAY_SIZE,
    parameter int DATA_W = DATA_WIDTH,
    parameter int ACC_W  = ACC_WIDTH
) (
    input  logic                          clk,
    input  logic                          rst_n,

    input  logic                          en,        // MAC enable broadcast
    input  logic                          clr,       // clear all N accumulators

    input  logic signed [DATA_W-1:0]      a_in,      // broadcast activation
    input  logic signed [N*DATA_W-1:0]    w_in,      // N packed weights

    output logic signed [N*ACC_W-1:0]     acc_out    // N packed accumulators
);

    genvar k;
    generate
        for (k = 0; k < N; k++) begin : g_pe
            logic signed [DATA_W-1:0] w_k;
            logic signed [ACC_W-1:0]  acc_k;
            assign w_k = w_in[k*DATA_W +: DATA_W];

            pe #(.DATA_W(DATA_W), .ACC_W(ACC_W)) u_pe (
                .clk   (clk),
                .rst_n (rst_n),
                .en    (en),
                .clr   (clr),
                .a_in  (a_in),
                .b_in  (w_k),
                .acc   (acc_k)
            );

            assign acc_out[k*ACC_W +: ACC_W] = acc_k;
        end
    endgenerate

endmodule
