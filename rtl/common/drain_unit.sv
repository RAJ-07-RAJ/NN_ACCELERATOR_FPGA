// =============================================================================
// drain_unit.sv -- read accumulators out of the PE array, one neuron per cycle.
//
// After a tile of N output neurons has finished accumulating, the controller
// raises `start` for one cycle. We then drive `idx = 0..N-1`, presenting the
// k-th accumulator on the data output along with the matching bias address.
//
// The PPU consumes this stream at 1 element/cycle.
// =============================================================================

`timescale 1ns/1ps

module drain_unit
    import nn_pkg::*;
#(
    parameter int N      = ARRAY_SIZE,
    parameter int ACC_W  = ACC_WIDTH,
    parameter int BSRAM_AW = nn_pkg::BSRAM_AW
) (
    input  logic                          clk,
    input  logic                          rst_n,

    input  logic                          start,        // 1-cycle pulse
    input  logic [15:0]                   tile_out_base,// first neuron idx of tile
    input  logic [15:0]                   tile_count,   // # neurons in this tile (<=N)

    input  logic signed [N*ACC_W-1:0]     accs,         // packed accumulators

    output logic                          out_valid,
    output logic signed [ACC_W-1:0]       out_acc,
    output logic [BSRAM_AW-1:0]           out_bias_addr,
    output logic                          busy,
    output logic                          done
);

    typedef enum logic [1:0] {IDLE, RUN, FIN} state_t;
    state_t state, nstate;

    logic [15:0] cnt;          // index inside the tile (0..tile_count-1)

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            cnt   <= '0;
        end else begin
            state <= nstate;
            if (state == IDLE && start)        cnt <= '0;
            else if (state == RUN)              cnt <= cnt + 16'd1;
        end
    end

    always_comb begin
        nstate = state;
        unique case (state)
            IDLE: if (start)                    nstate = RUN;
            RUN : if (cnt == tile_count - 1)    nstate = FIN;
            FIN :                               nstate = IDLE;
            default:                            nstate = IDLE;
        endcase
    end

    // Output mux: select acc[cnt]
    logic signed [ACC_W-1:0] acc_sel;
    always_comb begin
        acc_sel = '0;
        for (int i = 0; i < N; i++) begin
            if (i == cnt[$clog2(N>1?N:2)-1:0]) acc_sel = accs[i*ACC_W +: ACC_W];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid     <= 1'b0;
            out_acc       <= '0;
            out_bias_addr <= '0;
        end else begin
            out_valid     <= (state == RUN);
            out_acc       <= acc_sel;
            out_bias_addr <= tile_out_base[BSRAM_AW-1:0] + cnt[BSRAM_AW-1:0];
        end
    end

    assign busy = (state != IDLE);
    assign done = (state == FIN);

endmodule
