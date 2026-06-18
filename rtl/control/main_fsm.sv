// =============================================================================
// main_fsm.sv -- 7-state top-level controller.
//
// States:
//     IDLE            : wait for `start` from CSR.
//     LOAD_WEIGHTS    : DMA all weights and biases from DRAM into BRAMs.
//                       (One-time, can be skipped on subsequent inferences
//                       by leaving weights_loaded high.)
//     LOAD_INPUT      : DMA the input image from DRAM into Activation SRAM A.
//     COMPUTE_LAYER1  : run fc1 (IN=784, OUT=128, ReLU). Source = A, dest = B.
//     COMPUTE_LAYER2  : run fc2 (IN=128, OUT=64,  ReLU). Source = B, dest = A.
//     COMPUTE_OUTPUT  : run fc3 (IN=64,  OUT=10,  no ReLU). Source = A, dest = O.
//     WRITE_OUTPUT    : (optional) DMA Output SRAM back to DRAM.
//     DONE            : pulse done, return to IDLE.
//
// COMPUTE_LAYERx all share one "compute layer" datapath driven by a small
// inner controller (compute_ctrl, below) that:
//     for tile in 0 .. ceil(OUT/N)-1:
//         clr accumulators
//         for k in 0 .. IN-1:
//             present a[k]  +  W[tile, k, :]   to PE array (en=1)
//         start drain unit -> PPU -> dest SRAM
//
// Cycle estimate (ARRAY_SIZE = N):
//     layer_cycles ≈ ceil(OUT/N) * (IN + N + PIPELINE_BUBBLES)
//     For N=16:
//         fc1: 8 tiles * (784 + 16 + 6)  =  6,448 cycles
//         fc2: 4 tiles * (128 + 16 + 6)  =    600 cycles
//         fc3: 1 tile  * ( 64 + 10 + 6)  =     80 cycles
//     Total compute ≈ 7,128 cycles ~ 71 us @ 100 MHz.
// =============================================================================

`timescale 1ns/1ps

module main_fsm
    import nn_pkg::*;
(
    input  logic        clk,
    input  logic        rst_n,

    // CSR
    input  logic        start,
    output logic        busy,
    output logic        done,

    // compute-layer datapath handshake
    output logic [1:0]  cl_layer_id,    // 0=fc1,1=fc2,2=fc3
    output logic        cl_start,
    input  logic        cl_busy,
    input  logic        cl_done,

    // DMA handshake (shared for weights/input)
    output logic        dma_start,
    output logic [1:0]  dma_dst_sel,    // 0=W,1=A,2=B,3=Bias
    output logic [31:0] dma_src_addr,
    output logic [31:0] dma_dst_addr,
    output logic [31:0] dma_length,
    input  logic        dma_busy,
    input  logic        dma_done,

    // src/dst bank pointers for the 3 compute layers
    output logic        src_bank_l1, dst_bank_l1,
    output logic        src_bank_l2, dst_bank_l2,
    output logic        src_bank_l3, dst_bank_l3,

    // input ptr from CSR (where in "DRAM" the image lives)
    input  logic [31:0] input_dram_ptr
);

    typedef enum logic [3:0] {
        S_IDLE,
        S_LOAD_W,
        S_LOAD_B,
        S_LOAD_IN,
        S_COMP_L1,
        S_COMP_L2,
        S_COMP_L3,
        S_WRITE_OUT,
        S_DONE
    } state_t;

    state_t state, nstate;
    logic   weights_loaded;

    // ---- bank selection -----------------------------------------------------
    // Ping-pong:
    //   load   -> A
    //   fc1: A -> B
    //   fc2: B -> A
    //   fc3: A -> O (output sram)
    assign src_bank_l1 = 1'b0; assign dst_bank_l1 = 1'b1;
    assign src_bank_l2 = 1'b1; assign dst_bank_l2 = 1'b0;
    assign src_bank_l3 = 1'b0; assign dst_bank_l3 = 1'b0; // fc3 writes to OSRAM, bank irrelevant

    // ---- FSM ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            weights_loaded <= 1'b0;
        end else begin
            state <= nstate;
            if (state == S_LOAD_B && dma_done) weights_loaded <= 1'b1;
        end
    end

    always_comb begin
        nstate = state;
        unique case (state)
            S_IDLE      : if (start) begin
                              if (weights_loaded) nstate = S_LOAD_IN;
                              else                nstate = S_LOAD_W;
                          end
            S_LOAD_W    : if (dma_done)         nstate = S_LOAD_B;
            S_LOAD_B    : if (dma_done)         nstate = S_LOAD_IN;
            S_LOAD_IN   : if (dma_done)         nstate = S_COMP_L1;
            S_COMP_L1   : if (cl_done)          nstate = S_COMP_L2;
            S_COMP_L2   : if (cl_done)          nstate = S_COMP_L3;
            S_COMP_L3   : if (cl_done)          nstate = S_DONE;
            S_DONE      :                       nstate = S_IDLE;
            default     :                       nstate = S_IDLE;
        endcase
    end

    // ---- DMA programming ----------------------------------------------------
    // We use a simple flat DRAM map for the testbench:
    //   0x0000_0000 .. : packed weights (WSRAM_DEPTH words)
    //   0x0001_0000 .. : biases (sum of out sizes, INT32)
    //   input_dram_ptr  : the image (INPUT_SIZE bytes)
    localparam logic [31:0] DRAM_W_BASE = 32'h0000_0000;
    localparam logic [31:0] DRAM_B_BASE = 32'h0001_0000;
    localparam int          TOTAL_BIAS  = HIDDEN1_SIZE + HIDDEN2_SIZE + OUTPUT_SIZE;

    always_comb begin
        dma_start    = 1'b0;
        dma_dst_sel  = 2'd0;
        dma_src_addr = '0;
        dma_dst_addr = '0;
        dma_length   = '0;

        unique case (state)
            S_LOAD_W: begin
                dma_start    = (!dma_busy && !dma_done);
                dma_dst_sel  = 2'd0;             // WSRAM
                dma_src_addr = DRAM_W_BASE;
                dma_dst_addr = 32'd0;
                dma_length   = WSRAM_DEPTH;
            end
            S_LOAD_B: begin
                dma_start    = (!dma_busy && !dma_done);
                dma_dst_sel  = 2'd3;             // BSRAM
                dma_src_addr = DRAM_B_BASE;
                dma_dst_addr = 32'd0;
                dma_length   = TOTAL_BIAS;
            end
            S_LOAD_IN: begin
                dma_start    = (!dma_busy && !dma_done);
                dma_dst_sel  = 2'd1;             // ASRAM bank A
                dma_src_addr = input_dram_ptr;
                dma_dst_addr = 32'd0;
                dma_length   = INPUT_SIZE;
            end
            default: ;
        endcase
    end

    // The compute layer launcher is edge-triggered: pulse cl_start for one
    // cycle on entry into the COMP_* state.
    logic comp_started;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) comp_started <= 1'b0;
        else begin
            if (state != nstate && (nstate == S_COMP_L1 || nstate == S_COMP_L2 || nstate == S_COMP_L3))
                comp_started <= 1'b0;
            else if ((state == S_COMP_L1 || state == S_COMP_L2 || state == S_COMP_L3) && !cl_busy)
                comp_started <= 1'b1;
        end
    end

    assign cl_start    = (state == S_COMP_L1 || state == S_COMP_L2 || state == S_COMP_L3)
                          && !comp_started && !cl_busy;
    assign cl_layer_id = (state == S_COMP_L1) ? 2'd0
                       : (state == S_COMP_L2) ? 2'd1
                       : (state == S_COMP_L3) ? 2'd2 : 2'd0;

    assign busy = (state != S_IDLE) && (state != S_DONE);
    assign done = (state == S_DONE);

endmodule
