// =============================================================================
// compute_layer.sv -- inner controller that runs ONE FC layer to completion.
//
// Pipeline (no separate stagger module needed -- alignment is handled
// explicitly here):
//
//   cycle T  : FSM issues read of activation[k] and weight[tile,k] from BRAM.
//              (Activation/Weight SRAMs are read-first registered, so the
//               data is valid on the *output* port at the END of T,
//               i.e. visible to combinational logic at T+1.)
//   cycle T+1: PE samples (a, w), multiplies, and accumulates.
//
// So the en signal to the PE must be high one cycle AFTER the read was
// issued.  We implement this by simply doing both in the L_MAC state:
//     - on entering L_MAC we issue read for k=0    (no MAC yet)
//     - on the SECOND cycle we issue read for k=1 AND MAC k=0
//     - ...
//     - last cycle we issue no read but MAC the in-flight data
//
// We add an explicit FIRST_MAC bubble cycle so the implementation matches
// the description above exactly.
//
// For each layer:
//   for tile in 0 .. ceil(out_size / N) - 1:
//       L_TILE_INIT  : assert clr to PE array, issue read of (a[0], w[tile,0])
//       L_MAC        : for k=1..IN-1: MAC previous (a[k-1],w[k-1]), issue read of k
//       L_MAC_TAIL   : MAC the last in-flight (a[IN-1],w[IN-1])
//       L_DRAIN_WAIT : 1-cycle gap so acc settles
//       L_DRAIN      : drain N accs into PPU
//       L_TILE_DONE  : advance tile, loop
//   L_FIN            : signal done
// =============================================================================

`timescale 1ns/1ps

module compute_layer
    import nn_pkg::*;
#(
    parameter int N      = ARRAY_SIZE,
    parameter int DATA_W = DATA_WIDTH,
    parameter int ACC_W  = ACC_WIDTH
) (
    input  logic                          clk,
    input  logic                          rst_n,

    // run control
    input  logic                          start,
    output logic                          busy,
    output logic                          done,

    // layer parameters (latched on start)
    input  logic [15:0]                   in_size,
    input  logic [15:0]                   out_size,
    input  logic                          relu_en,
    input  logic [REQ_WIDTH-1:0]          m_q,
    input  logic [WSRAM_AW-1:0]           w_base,
    input  logic [BSRAM_AW-1:0]           b_base,
    input  logic                          src_bank,        // 0=A, 1=B
    input  logic                          dst_bank,        // 0=A, 1=B
    input  logic                          dst_is_output,   // override -> OSRAM

    // activation SRAM A
    output logic                          a_re, a_we,
    output logic [ASRAM_AW-1:0]           a_addr,
    output logic [DATA_W-1:0]             a_din,
    input  logic [DATA_W-1:0]             a_dout,

    // activation SRAM B
    output logic                          b_re, b_we,
    output logic [ASRAM_AW-1:0]           b_addr,
    output logic [DATA_W-1:0]             b_din,
    input  logic [DATA_W-1:0]             b_dout,

    // weight SRAM
    output logic                          w_re,
    output logic [WSRAM_AW-1:0]           w_addr,
    input  logic [WSRAM_WORD_W-1:0]       w_dout,

    // bias SRAM
    output logic                          bias_re,
    output logic [BSRAM_AW-1:0]           bias_addr,
    input  logic [BIAS_WIDTH-1:0]         bias_dout,

    // output SRAM
    output logic                          o_we,
    output logic [OSRAM_AW-1:0]           o_addr,
    output logic [DATA_W-1:0]             o_din
);

    // ---- latched layer params ----------------------------------------------
    logic [15:0]            in_size_r, out_size_r;
    logic                   relu_r;
    logic [REQ_WIDTH-1:0]   m_q_r;
    logic [WSRAM_AW-1:0]    w_base_r;
    logic [BSRAM_AW-1:0]    b_base_r;
    logic                   src_b_r, dst_b_r, dst_out_r;

    // ---- state -------------------------------------------------------------
    typedef enum logic [3:0] {
        L_IDLE, L_TILE_INIT, L_MAC, L_MAC_TAIL, L_DRAIN_WAIT,
        L_DRAIN, L_DRAIN_TAIL, L_TILE_DONE, L_FLUSH, L_FIN
    } state_t;
    state_t state, nstate;

    logic [15:0] tile_idx;        // 0..ceil(out/N)-1
    logic [15:0] tile_base;       // tile_idx * N
    logic [15:0] tile_cnt;        // neurons in this tile (<=N)
    logic [15:0] k;               // input index 0..in_size-1
    logic [15:0] num_tiles;

    assign num_tiles = (out_size_r + N - 1) / N;

    // ---- weight address calculation ----------------------------------------
    // Layout per quantize.py / pack_mem.py:
    //   for tile in 0..num_tiles-1:
    //     for k in 0..in_size-1:
    //       word at (w_base + tile*in_size + k) holds {W[tile*N+N-1, k], ..., W[tile*N, k]}
    logic [WSRAM_AW-1:0] w_addr_calc;
    assign w_addr_calc = w_base_r + tile_idx * in_size_r + k;

    // ---- src activation mux -------------------------------------------------
    logic [DATA_W-1:0] act_dout;
    assign act_dout = src_b_r ? b_dout : a_dout;

    // ---- FSM ---------------------------------------------------------------
    // Drain index (separate from k)
    logic [15:0] drain_cnt;
    logic        drain_in_progress;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= L_IDLE;
            tile_idx    <= '0;
            tile_base   <= '0;
            tile_cnt    <= '0;
            k           <= '0;
            drain_cnt   <= '0;
            in_size_r   <= '0; out_size_r <= '0;
            relu_r      <= 1'b0; m_q_r <= '0;
            w_base_r    <= '0; b_base_r <= '0;
            src_b_r     <= 1'b0; dst_b_r <= 1'b0; dst_out_r <= 1'b0;
        end else begin
            state <= nstate;

            case (state)
                L_IDLE: if (start) begin
                    in_size_r  <= in_size;
                    out_size_r <= out_size;
                    relu_r     <= relu_en;
                    m_q_r      <= m_q;
                    w_base_r   <= w_base;
                    b_base_r   <= b_base;
                    src_b_r    <= src_bank;
                    dst_b_r    <= dst_bank;
                    dst_out_r  <= dst_is_output;
                    tile_idx   <= '0;
                    tile_base  <= '0;
                end
                L_TILE_INIT: begin
                    // First read (k=0) is issued combinationally this cycle.
                    // Reset k to 1 -- next cycle (L_MAC) will issue read for
                    // k=1 while PE MACs the result of the k=0 read.
                    k        <= 16'd1;
                    if ((tile_base + N) <= out_size_r) tile_cnt <= 16'(N);
                    else                                tile_cnt <= out_size_r - tile_base;
                    drain_cnt <= '0;
                end
                L_MAC: begin
                    if (k < in_size_r - 1) k <= k + 16'd1;
                end
                L_MAC_TAIL: begin
                    // nothing to bump
                end
                L_DRAIN_WAIT: begin
                    drain_cnt <= '0;
                end
                L_DRAIN: begin
                    drain_cnt <= drain_cnt + 16'd1;
                end
                L_DRAIN_TAIL: begin
                    // pipeline tail; nothing to do
                end
                L_TILE_DONE: begin
                    tile_idx  <= tile_idx + 16'd1;
                    tile_base <= tile_base + 16'(N);
                    k         <= '0;
                end
                default: ;
            endcase
        end
    end

    // Pre-compute branch to keep next-state combinational clean.
    logic last_tile;
    assign last_tile = (tile_idx + 16'd1 == num_tiles);

    always_comb begin
        nstate = state;
        case (state)
            L_IDLE       : if (start)                       nstate = L_TILE_INIT;
            L_TILE_INIT  : begin
                // If the layer has only one input (degenerate), skip MAC loop.
                if (in_size_r <= 16'd1) nstate = L_MAC_TAIL;
                else                    nstate = L_MAC;
            end
            L_MAC        : if (k == in_size_r - 1)          nstate = L_MAC_TAIL;
            L_MAC_TAIL   :                                  nstate = L_DRAIN_WAIT;
            L_DRAIN_WAIT :                                  nstate = L_DRAIN;
            L_DRAIN      : if (drain_cnt == tile_cnt - 1)   nstate = L_DRAIN_TAIL;
            L_DRAIN_TAIL :                                  nstate = L_TILE_DONE;
            L_TILE_DONE  : begin
                                if (last_tile)              nstate = L_FLUSH;
                                else                        nstate = L_TILE_INIT;
                          end
            L_FLUSH      : if (wr_neuron_idx == out_size_r) nstate = L_FIN;
            L_FIN        :                                  nstate = L_IDLE;
            default      :                                  nstate = L_IDLE;
        endcase
    end

    // ---- control pulses to PE ----------------------------------------------
    // clr fires on the FIRST L_TILE_INIT cycle: it resets the accumulator
    // exactly one cycle before the first MAC.
    logic pe_clr;
    logic pe_en;

    assign pe_clr = (state == L_TILE_INIT);
    // en is high during L_MAC and L_MAC_TAIL: in L_MAC we MAC the in-flight
    // (k-1) data, and in L_MAC_TAIL we MAC the final (in_size-1) data.
    assign pe_en  = (state == L_MAC) || (state == L_MAC_TAIL);

    // ---- SRAM read address generation --------------------------------------
    // We want the BRAM to deliver a[k] and w[tile,k] on the cycle when the
    // PE will MAC them.  Reads are 1-cycle: address presented at T, data
    // appears at the dout port at T+1.
    //
    // Schedule:
    //   T (L_TILE_INIT) : addr = 0    -> dout for k=0 visible at T+1
    //   T+1 (L_MAC k=1) : addr = 1    -> dout for k=1 visible at T+2
    //                     pe_en=1: PE MACs the k=0 data
    //   ...
    //   T+IN-1 (L_MAC k=IN-1)         -> addr = IN-1   (last useful read)
    //                                    PE MACs k=IN-2
    //   T+IN (L_MAC_TAIL)             -> no read needed
    //                                    PE MACs k=IN-1
    //
    // So the address presented during L_TILE_INIT is 0, during L_MAC is `k`
    // (which the FSM has bumped to 1, 2, ..., IN-1).  We never present k=0
    // in L_MAC.

    logic [ASRAM_AW-1:0] act_rd_addr;
    logic [WSRAM_AW-1:0] w_rd_addr;
    always_comb begin
        if (state == L_TILE_INIT) begin
            act_rd_addr = '0;
            w_rd_addr   = w_base_r + tile_idx * in_size_r;
        end else begin
            act_rd_addr = k[ASRAM_AW-1:0];
            w_rd_addr   = w_addr_calc;
        end
    end

    // ---- PPU output writeback counter --------------------------------------
    // PPU has a 3-stage pipeline; valid stream from drain comes out (1 + 3)
    // cycles later than drain_cnt.  We don't need to know the absolute
    // delay -- we just consume PPU's out_valid stream and write into the
    // destination one address at a time.
    logic [15:0] wr_neuron_idx;
    logic        wr_valid_ppu;
    logic [DATA_W-1:0] wr_data_ppu;
    logic        wr_in_layer;     // 1 while we're writing the current layer

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_neuron_idx <= '0;
            wr_in_layer   <= 1'b0;
        end else begin
            if (state == L_IDLE && nstate == L_TILE_INIT) begin
                // entering a new layer run
                wr_neuron_idx <= '0;
                wr_in_layer   <= 1'b1;
            end else if (state == L_FIN || (state == L_IDLE && wr_in_layer && wr_neuron_idx == out_size_r)) begin
                wr_in_layer   <= 1'b0;
            end else if (wr_valid_ppu && wr_in_layer) begin
                wr_neuron_idx <= wr_neuron_idx + 16'd1;
            end
        end
    end

    // ---- SRAM port drives --------------------------------------------------
    always_comb begin
        // Default: idle
        a_re = 1'b0; a_we = 1'b0; a_addr = '0; a_din = '0;
        b_re = 1'b0; b_we = 1'b0; b_addr = '0; b_din = '0;
        w_re = 1'b0; w_addr = '0;
        o_we = 1'b0; o_addr = '0; o_din = '0;

        // Read activation from src bank
        if ((state == L_TILE_INIT) || (state == L_MAC)) begin
            if (src_b_r == 1'b0) begin
                a_re   = 1'b1;
                a_addr = act_rd_addr;
            end else begin
                b_re   = 1'b1;
                b_addr = act_rd_addr;
            end
            // Read weight
            w_re   = 1'b1;
            w_addr = w_rd_addr;
        end

        // Write PPU output to dst (activation bank OR output SRAM)
        if (wr_valid_ppu && wr_in_layer) begin
            if (dst_out_r) begin
                o_we   = 1'b1;
                o_addr = wr_neuron_idx[OSRAM_AW-1:0];
                o_din  = wr_data_ppu;
            end else if (dst_b_r == 1'b0) begin
                a_we   = 1'b1;
                a_addr = wr_neuron_idx[ASRAM_AW-1:0];
                a_din  = wr_data_ppu;
            end else begin
                b_we   = 1'b1;
                b_addr = wr_neuron_idx[ASRAM_AW-1:0];
                b_din  = wr_data_ppu;
            end
        end
    end

    // ---- PE array ----------------------------------------------------------
    logic signed [N*ACC_W-1:0] accs;

    pe_array #(.N(N), .DATA_W(DATA_W), .ACC_W(ACC_W)) u_array (
        .clk(clk), .rst_n(rst_n),
        .en (pe_en),
        .clr(pe_clr),
        .a_in(act_dout),
        .w_in(w_dout),
        .acc_out(accs)
    );

    // ---- drain unit --------------------------------------------------------
    logic                       drain_valid;
    logic signed [ACC_W-1:0]    drain_acc;
    logic [BSRAM_AW-1:0]        drain_bias_addr;

    // We trigger the drain by a one-cycle pulse: when entering L_DRAIN we
    // begin presenting accumulators.  In this rewrite we drive the drain
    // mux directly from the FSM's drain_cnt:

    logic signed [ACC_W-1:0] acc_sel;
    always_comb begin
        acc_sel = '0;
        for (int i = 0; i < N; i++) begin
            if (i == drain_cnt[$clog2(N>1?N:2)-1:0]) acc_sel = accs[i*ACC_W +: ACC_W];
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            drain_valid     <= 1'b0;
            drain_acc       <= '0;
            drain_bias_addr <= '0;
        end else begin
            drain_valid     <= (state == L_DRAIN);
            drain_acc       <= acc_sel;
            drain_bias_addr <= tile_base[BSRAM_AW-1:0] + drain_cnt[BSRAM_AW-1:0];
        end
    end

    // ---- bias SRAM read ----------------------------------------------------
    assign bias_re   = 1'b1;
    assign bias_addr = b_base_r + drain_bias_addr;

    // Pipeline alignment for PPU:
    //   path A (acc) : drain_cnt -> acc_sel (combo) -> drain_acc (1 reg) -> drain_acc_d1 (1 reg)  = 2 cycles
    //   path B (bias): drain_cnt -> drain_bias_addr (1 reg) -> bias_addr (combo) -> BRAM (1 cyc) -> bias_dout = 2 cycles
    //
    // Both arrive together if we register the acc-side one extra cycle.
    logic signed [BIAS_WIDTH-1:0] bias_q;
    logic                          drain_valid_q, drain_valid_d1;
    logic signed [ACC_W-1:0]       drain_acc_q, drain_acc_d1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bias_q          <= '0;
            drain_valid_q   <= 1'b0;
            drain_valid_d1  <= 1'b0;
            drain_acc_q     <= '0;
            drain_acc_d1    <= '0;
        end else begin
            drain_acc_d1    <= drain_acc;          // extra align stage
            drain_valid_d1  <= drain_valid;
            drain_acc_q     <= drain_acc_d1;
            drain_valid_q   <= drain_valid_d1;
            bias_q          <= bias_dout;           // 1 cycle after addr presented
        end
    end

    // ---- PPU ---------------------------------------------------------------
    ppu u_ppu (
        .clk(clk), .rst_n(rst_n),
        .in_valid (drain_valid_q),
        .in_acc   (drain_acc_q),
        .in_bias  (bias_q),
        .m_q      (m_q_r),
        .relu_en  (relu_r),
        .out_valid(wr_valid_ppu),
        .out_data (wr_data_ppu)
    );

    // ---- handshakes --------------------------------------------------------
    // done pulses at L_FIN; we hold until the PPU has flushed its last write,
    // so we wait one extra cycle after the final PPU output to declare done.
    // Simplification: wait until wr_neuron_idx == out_size_r.
    logic finish_pending;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) finish_pending <= 1'b0;
        else if (state == L_FIN && wr_neuron_idx == out_size_r) finish_pending <= 1'b1;
        else if (finish_pending) finish_pending <= 1'b0;
    end

    // busy = anywhere in the run; done = pulse when finished writing all outputs
    assign busy = (state != L_IDLE) || wr_in_layer;
    assign done = (state == L_FIN);

endmodule
