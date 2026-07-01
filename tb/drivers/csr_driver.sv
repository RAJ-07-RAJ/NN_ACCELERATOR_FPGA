// =============================================================================
// csr_driver.sv -- procedural driver for the simple CSR bus
//
// Exposes tasks that the top TB can call to issue reads/writes without
// duplicating handshake code.  Modelled as a small interface + bind.
// =============================================================================
`timescale 1ns/1ps

interface csr_drv_if (input logic clk);
    logic        we;
    logic [7:0]  addr;
    logic [31:0] wdata;
    logic [31:0] rdata;
endinterface

// Reusable task library
package csr_driver_pkg;
    task automatic csr_write(virtual csr_drv_if vif,
                             input logic [7:0] addr,
                             input logic [31:0] data);
        @(posedge vif.clk);
        vif.we    <= 1'b1;
        vif.addr  <= addr;
        vif.wdata <= data;
        @(posedge vif.clk);
        vif.we    <= 1'b0;
    endtask

    task automatic csr_read(virtual csr_drv_if vif,
                            input  logic [7:0] addr,
                            output logic [31:0] data);
        @(posedge vif.clk);
        vif.we   <= 1'b0;
        vif.addr <= addr;
        @(posedge vif.clk);
        data = vif.rdata;
    endtask

    task automatic poll_until_done(virtual csr_drv_if vif,
                                   input int max_polls = 100000);
        logic [31:0] status;
        int i;
        for (i = 0; i < max_polls; i++) begin
            csr_read(vif, 8'h04, status);
            if (status[1]) return;
        end
        $error("poll_until_done: timeout after %0d polls", max_polls);
    endtask
endpackage
