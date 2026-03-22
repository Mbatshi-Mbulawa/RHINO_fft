`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/02/2026 04:08:54 PM
// Design Name: 
// Module Name: axis_tlast_gen
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module axis_tlast_gen #
(
    parameter integer DATA_WIDTH = 128,
    parameter integer FRAME_LEN  = 16384
)
(
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis:m_axis, ASSOCIATED_RESET aresetn, CLK_DOMAIN system_overlay_zynq_ultra_ps_e_0_0_pl_clk0, FREQ_HZ 99999985" *)
    input  wire                    aclk,

    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire                    aresetn,

    // Slave AXIS (from RFDC)
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TDATA" *)
    input  wire [DATA_WIDTH-1:0]   s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TVALID" *)
    input  wire                    s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TREADY" *)
    output wire                    s_axis_tready,

    // Master AXIS (to AXI DMA S2MM)
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TDATA" *)
    output reg  [DATA_WIDTH-1:0]   m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TVALID" *)
    output reg                     m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TREADY" *)
    input  wire                    m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TLAST" *)
    output reg                     m_axis_tlast
);

    reg [$clog2(FRAME_LEN)-1:0] sample_cnt;

    assign s_axis_tready = m_axis_tready;  // simple pass-through backpressure

    always @(posedge aclk) begin
        if (!aresetn) begin
            sample_cnt    <= 0;
            m_axis_tdata  <= {DATA_WIDTH{1'b0}};
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;
        end else begin
            m_axis_tlast <= 1'b0;  // default
            if (s_axis_tvalid && s_axis_tready) begin
                m_axis_tdata  <= s_axis_tdata;
                m_axis_tvalid <= 1'b1;
                if (sample_cnt == FRAME_LEN-1) begin
                    m_axis_tlast <= 1'b1;
                    sample_cnt   <= 0;
                end else begin
                    sample_cnt <= sample_cnt + 1'b1;
                end
            end else begin
                m_axis_tvalid <= 1'b0;
            end
        end
    end

endmodule