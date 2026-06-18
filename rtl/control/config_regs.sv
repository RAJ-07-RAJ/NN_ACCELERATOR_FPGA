// =============================================================================
// config_regs.sv -- minimal AXI-lite-style memory-mapped CSR block.
//
// Register map (32-bit aligned, byte addresses):
//     0x00  CTRL          [0]=start (W1P), [1]=soft_reset (W1P)
//     0x04  STATUS        [0]=busy, [1]=done (RO, latched until cleared by W1C)
//     0x08  INPUT_PTR     base addr of input vector in DRAM
//     0x0C  OUTPUT_PTR    base addr where output should be written (unused in
//                         the BRAM-only sim, present for AXI extension)
//     0x10  IRQ_EN        [0] = enable done interrupt
//     0x14  IRQ_STATUS    [0] = done pending (W1C)
//     0x18  VERSION       RO, 32'h0000_0100
//     0x1C  CYCLE_COUNT   RO, free-running while busy
//
// We use a tiny *internal* style bus rather than the full AXI4-Lite to keep
// the example self-contained.  An axi_lite -> internal bridge is trivial to
// add later (one always_ff per channel).
// =============================================================================

`timescale 1ns/1ps

module config_regs (
    input  logic        clk,
    input  logic        rst_n,

    // simple register bus (host side)
    input  logic        reg_we,
    input  logic [7:0]  reg_addr,
    input  logic [31:0] reg_wdata,
    output logic [31:0] reg_rdata,

    // to/from main FSM
    output logic        start_pulse,
    output logic        soft_reset,
    input  logic        busy,
    input  logic        done,
    output logic [31:0] input_ptr,
    output logic [31:0] output_ptr,
    output logic        irq_en,
    output logic        irq          // level-high while pending
);

    logic [31:0] r_ctrl, r_status_done_latch;
    logic [31:0] r_input_ptr, r_output_ptr;
    logic [31:0] r_irq_en, r_irq_status;
    logic [31:0] r_cycles;

    logic        start_q, srst_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_ctrl              <= '0;
            r_status_done_latch <= '0;
            r_input_ptr         <= '0;
            r_output_ptr        <= '0;
            r_irq_en            <= '0;
            r_irq_status        <= '0;
            r_cycles            <= '0;
            start_q             <= 1'b0;
            srst_q              <= 1'b0;
        end else begin
            // pulse regs default low
            start_q <= 1'b0;
            srst_q  <= 1'b0;

            if (reg_we) begin
                unique case (reg_addr[7:2])
                    6'h0: begin
                        if (reg_wdata[0]) start_q <= 1'b1;
                        if (reg_wdata[1]) srst_q  <= 1'b1;
                    end
                    6'h1: begin
                        // W1C on STATUS.done
                        if (reg_wdata[1]) r_status_done_latch[1] <= 1'b0;
                    end
                    6'h2: r_input_ptr  <= reg_wdata;
                    6'h3: r_output_ptr <= reg_wdata;
                    6'h4: r_irq_en     <= {31'b0, reg_wdata[0]};
                    6'h5: if (reg_wdata[0]) r_irq_status <= '0;     // W1C
                    default: ;
                endcase
            end

            // latch done
            if (done) begin
                r_status_done_latch[1] <= 1'b1;
                r_irq_status[0]        <= 1'b1;
            end

            // cycle counter while busy
            if (busy) r_cycles <= r_cycles + 32'd1;
            else if (start_q) r_cycles <= '0;
        end
    end

    always_comb begin
        unique case (reg_addr[7:2])
            6'h0: reg_rdata = r_ctrl;
            6'h1: reg_rdata = {30'b0, r_status_done_latch[1], busy};
            6'h2: reg_rdata = r_input_ptr;
            6'h3: reg_rdata = r_output_ptr;
            6'h4: reg_rdata = r_irq_en;
            6'h5: reg_rdata = r_irq_status;
            6'h6: reg_rdata = 32'h0000_0100;
            6'h7: reg_rdata = r_cycles;
            default: reg_rdata = 32'h0;
        endcase
    end

    assign start_pulse = start_q;
    assign soft_reset  = srst_q;
    assign input_ptr   = r_input_ptr;
    assign output_ptr  = r_output_ptr;
    assign irq_en      = r_irq_en[0];
    assign irq         = r_irq_en[0] & r_irq_status[0];

endmodule
