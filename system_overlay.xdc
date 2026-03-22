# DAC Tile 228 reference clock - RFSoC 4x2 board
set_property PACKAGE_PIN G11 [get_ports dac0_clk_clk_p]
set_property PACKAGE_PIN F11 [get_ports dac0_clk_clk_n]
set_property IOSTANDARD LVDS [get_ports dac0_clk_clk_p]
set_property IOSTANDARD LVDS [get_ports dac0_clk_clk_n]

# ADC Tile 224 reference clock (already working but good to have explicit)
set_property PACKAGE_PIN T11 [get_ports adc0_clk_clk_p]
set_property PACKAGE_PIN T10 [get_ports adc0_clk_clk_n]
set_property IOSTANDARD LVDS [get_ports adc0_clk_clk_p]
set_property IOSTANDARD LVDS [get_ports adc0_clk_clk_n]