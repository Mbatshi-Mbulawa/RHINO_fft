# Complete Vivado Block Diagram Guide - RHINO RFSoC System
## From Zero to Working Bitstream

**Author:** Mbatshi Jerry Junior Mbulawa  
**Date:** February 27, 2025  
**Vivado Version:** 2025.x  
**Target:** RFSoC 4x2 Board  
**Goal:** 200 MSPS acquisition with 8192-point FFT

---

## Design Specifications

Based on supervisor requirements and Vlad's guidance:

| Parameter | Value | Reason |
|-----------|-------|--------|
| ADC Sample Rate | 3.2 GHz | Clean decimation: 3.2 GHz / 16 = 200 MSPS |
| Decimation Factor | 16x | Reduces to target 200 MSPS |
| Output Rate | 200 MSPS | PhD specification requirement |
| FFT Size | 8192 points | Proven, scientifically sufficient |
| Frequency Resolution | 24.4 kHz/bin | 200 MHz / 8192 |
| RHINO Band Coverage | 60-85 MHz | ~1025 bins (excellent coverage) |

---

## Part 1: Create New Project from Scratch

### Step 1.1: Delete Old Project (If Exists)

1. Close Vivado if open
2. Navigate to your project folder
3. **Delete the entire project folder**
4. Create fresh folder: `RHINO_RFSoC_200MSPS`

---

### Step 1.2: Create New Vivado Project

1. **Launch Vivado 2025.x**

2. Click **"Create Project"**

3. **Project Name:**
   - Name: `RHINO_RFSoC_200MSPS`
   - Location: Choose your work directory
   - ✅ Check: **"Create project subdirectory"**
   - Click **Next**

4. **Project Type:**
   - Select: **"RTL Project"**
   - ✅ Check: **"Do not specify sources at this time"**
   - Click **Next**

5. **Default Part:**
   - Click **"Boards"** tab
   - In search box, type: `RFSoC 4x2`
   - Select: **"RFSoC 4x2"**
   - If you don't see it, you need to install board files first
   - Click **Next**

6. **Summary:**
   - Review settings
   - Click **Finish**

---

### Step 1.3: Create Block Design

1. In **Flow Navigator** (left panel), under **IP INTEGRATOR**
2. Click **"Create Block Design"**
3. **Design name:** `rfsoc_rhino_fft` (lowercase, no spaces)
4. **Directory:** `<Local to Project>`
5. Click **OK**

You now have a **blank canvas** with a grid. Time to add IP blocks!

---

## Part 2: Add and Configure Zynq UltraScale+ MPSoC

### Step 2.1: Add Zynq Block

1. Click the **"+" icon** in the Diagram window (or press Ctrl+I)
2. Search: `zynq ultra`
3. Double-click: **"Zynq UltraScale+ MPSoC"**
4. Block appears on canvas
5. Click **OK** to close IP catalog

---

### Step 2.2: Run Block Automation

**CRITICAL:** This will auto-configure the Zynq for RFSoC 4x2 board

1. Look for green banner at top: **"Run Block Automation"**
2. Click **"Run Block Automation"**
3. A dialog appears
4. ✅ Check: **"Apply Board Preset"**
5. Click **OK**

Vivado will configure:
- DDR4 memory
- Clocks
- Resets
- Peripherals (Ethernet, USB, UART)

**Wait for it to complete** (5-10 seconds)

---

### Step 2.3: Fix DDR4 Row Address Error

**IMPORTANT:** The board preset has a bug. We must fix it manually.

1. **Double-click** the `zynq_ultra_ps_e_0` block
2. Navigate to: **"PS-PL Configuration"** (left menu)
3. Expand: **"PS-PL Interfaces"**
4. Expand: **"Master Interface"**
5. ✅ Ensure these are checked:
   - **M_AXI_HPM0_FPD** (CPU master 0)
   - **M_AXI_HPM1_FPD** (CPU master 1)

6. Expand: **"Slave Interface"**
7. ✅ Check: **S_AXI_HP0_FPD** (DMA will write here)

8. Navigate to: **"DDR Configuration"** (left menu)
9. Click: **"DDR Controller Configuration"**
10. Find: **"Row Address Count"**
11. Change from `17` to `16` ← **CRITICAL FIX**
12. Click **OK**

---

### Step 2.4: Verify Zynq Configuration

Check these settings are correct:

**Page: Clock Configuration**
- `pl_clk0`: Should be ~99-100 MHz (PS fabric clock)

**Page: PS-PL Configuration → PS-PL Interfaces**
- Master: M_AXI_HPM0_FPD ✓
- Master: M_AXI_HPM1_FPD ✓
- Slave: S_AXI_HP0_FPD ✓

**Page: DDR Configuration**
- Row Address Count: 16 ✓

Click **OK** to close configuration.

---

## Part 3: Add and Configure RF Data Converter

### Step 3.1: Add RF Data Converter Block

1. Press **Ctrl+I** (or click + icon)
2. Search: `rf data converter`
3. Double-click: **"Usp RF Data Converter"**
4. Block appears on canvas
5. Click **OK**

---

### Step 3.2: Configure RF Data Converter

**Double-click** the `usp_rf_data_converter_0` block

---

#### Page 1: ADC Settings

1. Click **"ADC"** tab (top)

2. **Tile 224 → ADC 0:**
   - ✅ Check: **"Enable Data Converter"**
   - **Sampling Rate:** `3200.000` (3.2 GHz)
   - **Type:** `0` (Real)
   - **Mixer Type:** `Bypassed` (direct sampling, no mixing)
   - **Decimation Mode:** `1x` for now (we'll set 16x in next step)

3. **ADC 1, 2, 3:** Leave unchecked (we only need ADC 0)

---

#### Page 2: Advanced ADC Settings

1. Click **"ADC"** tab still selected
2. Click **"Advanced"** sub-tab at bottom

3. **Tile 224 → ADC 0:**
   - **Decimation Factor:** Change to `16x` ← **KEY SETTING**
   - **Output Sample Rate:** Should automatically show `200.000` MSPS

4. Verify calculation:
   - 3200 MHz / 16 = 200 MHz ✓

---

#### Page 3: Clock Settings

1. Click **"Clocking"** tab (top)

2. **ADC Clock Distribution:**
   - **Tile 224 Clock Source:** `Internal PLL`
   - **PLL Mode:** `Enabled`
   - **PLL Reference Clock Frequency:** `204.8` MHz (RFSoC 4x2 reference)

3. **Fabric Clock (clk_adc0 output):**
   - **ADC Data Clock:** Check the frequency shown
   - Should be: `200.000` MHz (matches output rate after decimation)
   - If different, note the value — you'll use it later

4. **AXI-Lite Clock:**
   - Leave as default

---

#### Page 4: System Clocking

1. Click **"System Clocking"** tab

2. **Tile 224:**
   - **Sampling Rate:** Verify shows `3200.000` MHz
   - **Decimation:** Verify shows `16`
   - **Output Rate:** Verify shows `200.000` MSPS

3. **Fabric Clock Output:**
   - `clk_adc0`: Should show `200 MHz` or `250 MHz` (varies by config)
   - Note this frequency — you'll need it for Clock Wizard

4. Click **OK** to close configuration

---

### Step 3.3: Verify RF Data Converter Ports

After closing config, look at the `usp_rf_data_converter_0` block:

**Input Ports (left side):**
- `s_axi` — AXI-Lite control interface
- `s_axis_adc_*` — Won't use (we're using internal ADC)
- `adc0_clk` — **Reference clock input** (from Zynq PLL)

**Output Ports (right side):**
- `m00_axis` — **ADC sample output** (200 MSPS, goes to FFT)
- `clk_adc0` — **Output clock** (synchronous to ADC, ~200 MHz)
- `vin0_01_*` — Physical ADC input pins
- `irq` — Interrupt

---

## Part 4: Add AXI SmartConnect

This routes control traffic between Zynq CPU and IP blocks.

### Step 4.1: Add SmartConnect

1. Press **Ctrl+I**
2. Search: `smartconnect`
3. Double-click: **"AXI SmartConnect"**
4. Click **OK**

---

### Step 4.2: Configure SmartConnect

**Double-click** `axi_smc` block

1. **Number of Master Interfaces:** `2`
   - M00 → RF Data Converter control
   - M01 → FFT control (we'll add FFT next)

2. **Number of Slave Interfaces:** `2`
   - S00 → Zynq M_AXI_HPM0_FPD
   - S01 → Zynq M_AXI_HPM1_FPD

3. Click **OK**

---

## Part 5: Add FFT IP Core

### Step 5.1: Add FFT Block

1. Press **Ctrl+I**
2. Search: `fft`
3. Double-click: **"Fast Fourier Transform"** (xfft)
4. Click **OK**

---

### Step 5.2: Configure FFT

**Double-click** the `xfft_0` block

---

#### Page 1: Configuration

1. **Transform Length:** `8192`

2. **Implementation Options:**
   - **Architecture:** `Pipelined, Streaming I/O`
   - **Target Clock Frequency:** `250` MHz
   - **Target Data Throughput:** `50` MSPS (conservative, can be higher)

3. **Data Format:**
   - **Input Data Width:** `16` bits (matches RF Data Converter)
   - **Phase Factor Width:** `16` bits
   - **Output Data Width:** `16` bits (or check "Use Full Precision")

4. **Optional Fields:**
   - **ACLKEN:** Leave unchecked
   - **Optional Output Fields:** Check **TLast** ✓

5. **Scaling Options:**
   - **Scaling:** `Scaled` (prevents overflow)

6. Click **Next** → **Next** → view summary

7. Click **OK**

---

### Step 5.3: Verify FFT Ports

**Input Ports (left):**
- `S_AXIS_CONFIG` — Configuration input (optional, leave unconnected)
- `S_AXIS_DATA` — **Data input** (from RF Data Converter)
- `aclk` — **Clock input** (will come from Clock Wizard)

**Output Ports (right):**
- `M_AXIS_DATA` — **FFT output** (goes to DMA)
- `event_*` — Status signals

---

## Part 6: Add AXI DMA

### Step 6.1: Add DMA Block

1. Press **Ctrl+I**
2. Search: `axi dma`
3. Double-click: **"AXI Direct Memory Access"**
4. Click **OK**

---

### Step 6.2: Configure DMA

**Double-click** `axi_dma_0` block

1. **Enable Scatter Gather Engine:** ❌ Uncheck (simple mode)

2. **Enable Read Channel:** ❌ Uncheck (we only write to memory)

3. **Enable Write Channel:** ✅ Check

4. **Write Channel Configuration:**
   - **Stream Data Width:** `32` bits
   - **Max Burst Size:** `16`
   - **Buffer Length Register Width:** `23` bits (allows larger buffers)
   - **Address Width:** `32` bits

5. Click **OK**

---

### Step 6.3: Verify DMA Ports

**Slave Interfaces (left):**
- `S_AXI_LITE` — Control interface (for Python)
- `S_AXIS_S2MM` — **Data input** (from FFT)

**Master Interfaces (right):**
- `M_AXI_S2MM` — **Memory write interface** (to DDR via SmartConnect)

**Clocks:**
- `s_axi_lite_aclk` — Control clock
- `m_axi_s2mm_aclk` — Data clock

**Interrupts:**
- `s2mm_introut` — DMA completion interrupt

---

## Part 7: Add Clock Wizard (CRITICAL - Vlad's Solution)

This is the **KEY** to solving the clock issue!

### Step 7.1: Add Clock Wizard

1. Press **Ctrl+I**
2. Search: `clocking wizard`
3. Double-click: **"Clocking Wizard"**
4. Click **OK**

---

### Step 7.2: Configure Clock Wizard

**Double-click** `clk_wiz_0` block

---

#### Page 1: Clocking Features

1. **Primitive:** `MMCM` (Mixed-Mode Clock Manager)

2. **Frequency Synthesis:**
   - ✅ Enable

3. **Input Clock:**
   - **Port Name:** `clk_in1`
   - **Input Frequency:** Enter the frequency of RF Data Converter `clk_adc0` output
     - If RF DC showed 200 MHz → enter `200`
     - If RF DC showed 250 MHz → enter `250`

4. **Output Clocks:**
   - **clk_out1:**
     - ✅ Enable
     - **Output Freq (MHz):** `250` (for FFT and DMA)
     - **Requested:** `250.000`
     - **Actual:** Should match (if not, adjust slightly)

   - **clk_out2:**
     - ❌ Disable (not needed)

5. **Optional Inputs/Outputs:**
   - ✅ **locked** signal (indicates PLL is stable)
   - ✅ **reset** (active high)
   - **Reset Type:** `Active High`

6. Click **OK**

---

## Part 8: Add Processor System Reset Blocks

We need TWO reset blocks (one per clock domain).

### Step 8.1: Add First Reset Block

1. Press **Ctrl+I**
2. Search: `processor system reset`
3. Double-click: **"Processor System Reset"**
4. Rename: Right-click → Rename → `rst_clk_wiz_250M`
5. Click **OK**

---

### Step 8.2: Add Second Reset Block

1. Press **Ctrl+I**
2. Search: `processor system reset`
3. Double-click: **"Processor System Reset"**
4. Rename: Right-click → Rename → `rst_ps8_0_99M`
5. Click **OK**

---

## Part 9: Connect All IP Blocks (CRITICAL STEP)

### Connection Strategy:

We'll connect in this order:
1. Clocks first (prevents interface mismatches)
2. Resets second (ensures proper sequencing)
3. Control interfaces (AXI-Lite)
4. Data paths (AXI-Stream and AXI-MM)

---

### Step 9.1: Connect Zynq to RF Data Converter Reference Clock

**Goal:** Provide reference clock to RF Data Converter PLL

1. Find: `zynq_ultra_ps_e_0` → `pl_clk0` output (right side)
2. Find: `usp_rf_data_converter_0` → `adc0_clk` input (left side)

**Connection Method:**

If **direct drag works:**
3. Click and drag from `pl_clk0` to `adc0_clk`
4. Wire appears → Done ✓

If **"no matching connections" error:**
5. Right-click `pl_clk0` → **Make External**
6. Right-click `adc0_clk` → **Make External**
7. External ports appear on canvas edges
8. Manually rename both external ports to same name: `ref_clk`
9. They auto-connect
10. Right-click each external port → **Make Internal**

---

### Step 9.2: Connect RF Data Converter Output Clock to Clock Wizard

**CRITICAL:** This is Vlad's key insight!

1. Find: `usp_rf_data_converter_0` → `clk_adc0` **OUTPUT** (right side)
2. Find: `clk_wiz_0` → `clk_in1` input (left side)
3. **Drag** from `clk_adc0` to `clk_in1`

Wire appears. This ensures the processing clock is **frequency-locked** to the ADC!

---

### Step 9.3: Connect Clock Wizard Output to Processing Blocks

**Goal:** All data processing runs on Clock Wizard output (250 MHz)

1. Find: `clk_wiz_0` → `clk_out1` output (right side)

2. Connect to **FFT:**
   - Drag from `clk_out1` to `xfft_0` → `aclk`

3. Connect to **DMA:**
   - Drag from `clk_out1` to `axi_dma_0` → `m_axi_s2mm_aclk`
   - Drag from `clk_out1` to `axi_dma_0` → `s_axi_lite_aclk`

4. Connect to **First Reset Block:**
   - Drag from `clk_out1` to `rst_clk_wiz_250M` → `slowest_sync_clk`

---

### Step 9.4: Connect SmartConnect Clocks

**Goal:** SmartConnect needs clocks for each interface

1. Find: `zynq_ultra_ps_e_0` → `maxihpm0_fpd_aclk` output
   - Drag to `axi_smc` → `aclk` (main clock)

2. Connect **all** SmartConnect interface clocks to Zynq clocks:
   - `axi_smc` → `aclk0` ← `zynq` → `maxihpm0_fpd_aclk`
   - `axi_smc` → `aclk1` ← `zynq` → `maxihpm1_fpd_aclk`

3. Connect master side clocks:
   - `axi_smc` → `aclk` (already connected above)

---

### Step 9.5: Connect Zynq HP Slave Clock

**Goal:** DMA writes to Zynq memory, needs clock

1. Find: `zynq_ultra_ps_e_0` → `saxihp0_fpd_aclk` input (left side)
2. Find: `clk_wiz_0` → `clk_out1` output
3. **Drag** from `clk_out1` to `saxihp0_fpd_aclk`

---

### Step 9.6: Connect Second Reset Block Clock

1. Find: `zynq_ultra_ps_e_0` → `pl_clk0` output
2. Find: `rst_ps8_0_99M` → `slowest_sync_clk` input
3. **Drag** to connect

---

### Step 9.7: Connect Reset Signals

#### First Reset Block (ADC clock domain):

1. Find: `zynq_ultra_ps_e_0` → `pl_resetn0` output
2. Drag to: `rst_clk_wiz_250M` → `ext_reset_in`

3. Find: `clk_wiz_0` → `locked` output
4. Drag to: `rst_clk_wiz_250M` → `dcm_locked`

#### Second Reset Block (PS clock domain):

5. Find: `zynq_ultra_ps_e_0` → `pl_resetn0` output
6. Drag to: `rst_ps8_0_99M` → `ext_reset_in`

7. Find: `clk_wiz_0` → `locked` output (can branch wire)
8. Drag to: `rst_ps8_0_99M` → `dcm_locked`

---

### Step 9.8: Connect AXI-Lite Control Interfaces

**Goal:** CPU controls IP blocks via SmartConnect

#### Connect Zynq Masters to SmartConnect:

1. **Run Connection Automation** (green banner appears)
2. Select: `zynq_ultra_ps_e_0/M_AXI_HPM0_FPD`
3. Click **OK**

4. **Run Connection Automation** again
5. Select: `zynq_ultra_ps_e_0/M_AXI_HPM1_FPD`
6. Click **OK**

SmartConnect slave interfaces (S00, S01) are now connected.
**If the Run Connection Automation does not work, you can connect them automatically**

#### Connect SmartConnect Masters to IP Blocks:

7. **Manual connection** or **Run Connection Automation**:
   - `axi_smc` → `M00_AXI` → `usp_rf_data_converter_0` → `s_axi`
   - `axi_smc` → `M01_AXI` → `axi_dma_0` → `S_AXI_LITE`

If automation doesn't work:
- Drag manually
- Or right-click SmartConnect → **Customize Block** → increase masters

---

### Step 9.9: Connect Data Path (AXI-Stream)

**Goal:** ADC → FFT → DMA → Memory

#### ADC to FFT:

1. Find: `usp_rf_data_converter_0` → `m00_axis` output
2. Find: `xfft_0` → `S_AXIS_DATA` input
3. **Drag** to connect

#### FFT to DMA:

4. Find: `xfft_0` → `M_AXIS_DATA` output
5. Find: `axi_dma_0` → `S_AXIS_S2MM` input
6. **Drag** to connect

---

### Step 9.10: Connect DMA to Memory

**Goal:** DMA writes FFT data to DDR

1. **Run Connection Automation** (green banner)
2. Select: `axi_dma_0/M_AXI_S2MM`
3. Options:
   - **Master Interface:** `axi_smc` (goes through SmartConnect)
   - Click **OK**

Vivado auto-connects:
- `axi_dma_0` → `M_AXI_S2MM` → `axi_smc` → Zynq → DDR memory

---

### Step 9.11: Connect Reset Outputs to IP Blocks

#### ADC Clock Domain Resets:

1. Find: `rst_clk_wiz_250M` → `peripheral_aresetn` output
2. Connect to:
   - `xfft_0` → `aresetn`
   - `axi_dma_0` → `axi_resetn` (may be different name)

3. Find: `rst_clk_wiz_250M` → `interconnect_aresetn` output
4. Connect to:
   - `axi_smc` → `aresetn`

#### PS Clock Domain Resets:

5. Find: `rst_ps8_0_99M` → `peripheral_aresetn` output
6. Connect to any IP on PS clock (if any)

---

### Step 9.12: Connect Interrupt Signal

**Goal:** DMA notifies CPU when transfer complete

1. Find: `axi_dma_0` → `s2mm_introut` output
2. Find: `zynq_ultra_ps_e_0` → `pl_ps_irq0` input (may need to enable)

If `pl_ps_irq0` not visible:
3. **Double-click** Zynq block
4. **PS-PL Configuration** → **General** → **Interrupts**
5. ✅ Enable: **PL to PS IRQ0**
6. Click **OK**

7. Now connect: `axi_dma_0/s2mm_introut` → `zynq/pl_ps_irq0[0:0]`

---

### Step 9.13: Make External: Physical ADC Pins

**Goal:** Connect to external SMA antenna input

1. Find: `usp_rf_data_converter_0` → `vin0_01_*` ports
2. **Right-click** each → **Make External**
3. External ports appear on right edge

These will connect to physical pins defined in constraints.

---

### Step 9.14: Optional - Make External: Clock Wizard Reset

If you want manual reset control:

1. Find: `clk_wiz_0` → `reset` input
2. If not connected, right-click → **Make External**

Or connect to a constant:
3. Add: **Constant** IP (value = 0)
4. Connect: `constant` → `clk_wiz_0/reset`

---

## Part 10: Validate Design

### Step 10.1: Run Design Validation

1. Press **F6** (or **Tools → Validate Design**)
2. Vivado checks all connections

**Expected Result:** ✅ **"Validation successful"**

---

### Step 10.2: If Validation Fails

Check these common issues:

#### Clock Not Connected:
- Verify: `clk_adc0` → `clk_wiz_0/clk_in1` ✓
- Verify: `clk_wiz_0/clk_out1` → all processing blocks ✓

#### Reset Not Connected:
- Verify: All `aresetn` ports connected to reset blocks ✓

#### AXI Interface Mismatch:
- Check SmartConnect has enough master/slave ports
- Customize block if needed

#### Missing Interrupt:
- Enable PL to PS interrupts in Zynq config

---

### Step 10.3: Review Warnings (OK to Ignore)

These warnings are **safe to ignore**:

```
[BD 41-1306] The connection to interface pin </axi_smc/S00_AXI> 
is being overridden...
```
→ Normal, SmartConnect manages this

```
[BD 41-1629] </xfft_0/S_AXIS_CONFIG> is not connected
```
→ Optional input, we don't use runtime config

---

## Part 11: Assign Addresses

### Step 11.1: Open Address Editor

1. Click **"Address Editor"** tab (next to Diagram tab)
2. You should see:
   - `usp_rf_data_converter_0/s_axi`
   - `axi_dma_0/S_AXI_LITE`

---

### Step 11.2: Auto-Assign Addresses

1. Right-click in Address Editor
2. Select **"Assign All"**

Vivado assigns memory-mapped addresses automatically.

---

### Step 11.3: Verify Address Ranges

Check that addresses don't overlap:

```
RF Data Converter: 0xA000_0000 - 0xA000_FFFF
DMA:              0xA001_0000 - 0xA001_FFFF
```

(Actual values may differ, just verify no overlap)

---

## Part 12: Generate HDL Wrapper

### Step 12.1: Create Wrapper

1. In **Sources** window (bottom-left)
2. Find: **Design Sources** → `rfsoc_rhino_fft.bd`
3. **Right-click** → **Create HDL Wrapper**
4. Select: **"Let Vivado manage wrapper and auto-update"**
5. Click **OK**

Vivado creates: `rfsoc_rhino_fft_wrapper.v`

---

## Part 13: Run Synthesis

### Step 13.1: Start Synthesis

1. **Flow Navigator** → **SYNTHESIS**
2. Click **"Run Synthesis"**
3. Dialog appears:
   - **Number of jobs:** Use max (e.g., 8)
   - Click **OK**

**Estimated time:** 10-20 minutes

---

### Step 13.2: Wait for Synthesis

Progress bar appears. Vivado is:
- Converting block diagram to gates
- Optimizing logic
- Checking resource usage

**Go get coffee ☕**

---

### Step 13.3: Synthesis Complete

Dialog appears: **"Synthesis Completed Successfully"**

**Options:**
- ⚪ Run Implementation
- ⚪ Open Synthesized Design
- ⚪ View Reports

**Select:** Run Implementation  
**Click:** OK

---

## Part 14: Run Implementation

### Step 14.1: Start Implementation

Implementation runs automatically if you selected it above.

If not:
1. **Flow Navigator** → **IMPLEMENTATION**
2. Click **"Run Implementation"**
3. Click **OK**

**Estimated time:** 20-40 minutes

---

### Step 14.2: Wait for Implementation

Vivado is:
- Placing logic on physical FPGA locations
- Routing wires between logic
- Checking timing constraints

**Go get lunch 🍕**

---

### Step 14.3: Implementation Complete

Dialog appears: **"Implementation Completed Successfully"**

**Options:**
- ⚪ Generate Bitstream
- ⚪ Open Implemented Design
- ⚪ View Reports

**Select:** Generate Bitstream  
**Click:** OK

---

## Part 15: Generate Bitstream

### Step 15.1: Start Bitstream Generation

Bitstream generation runs automatically if selected above.

**Estimated time:** 5-15 minutes

---

### Step 15.2: Bitstream Complete

Dialog appears: **"Bitstream Generation Completed Successfully"**

**Congratulations! 🎉** You now have:
- `rfsoc_rhino_fft_wrapper.bit` (bitstream file)
- `rfsoc_rhino_fft_wrapper.hwh` (hardware handoff for PYNQ)

---

### Step 15.3: Locate Output Files

1. Navigate to:
```
<project_dir>/RHINO_RFSoC_200MSPS.runs/impl_1/
```

2. Find these files:
   - `rfsoc_rhino_fft_wrapper.bit`
   - `rfsoc_rhino_fft_wrapper.hwh`

3. **Copy both files** to a safe location (e.g., `~/bitstreams/`)

---

## Part 16: Check Resource Usage

### Step 16.1: Open Reports

1. **Flow Navigator** → **Open Implemented Design**
2. Click **"Report Utilization"**

---

### Step 16.2: Review Utilization

Check these resources:

| Resource | Used | Available | % |
|----------|------|-----------|---|
| LUTs | ~50K | 425K | ~12% |
| FFs | ~80K | 850K | ~9% |
| BRAM | ~100 | 1080 | ~9% |
| DSP | ~200 | 1248 | ~16% |

**If usage > 80% on any resource:** Design may have issues  
**If usage < 50%:** You have headroom for PFB or more features ✅

---

## Part 17: Export Hardware for PYNQ

### Step 17.1: Export XSA File

1. **File** → **Export** → **Export Hardware**
2. **Output:** `<Local to Project>`
3. ✅ Check: **"Include bitstream"**
4. Click **OK**

Vivado creates: `rfsoc_rhino_fft_wrapper.xsa`

This file contains **everything** PYNQ needs.

---

## Part 18: Transfer to RFSoC Board

### Step 18.1: Copy Files to Board

Use **SCP** (if board is on network):

```bash
# Copy bitstream
scp rfsoc_rhino_fft_wrapper.bit xilinx@<board-ip>:/home/xilinx/

# Copy hardware handoff
scp rfsoc_rhino_fft_wrapper.hwh xilinx@<board-ip>:/home/xilinx/
```

**Or** use **JupyterLab** upload:
1. Open JupyterLab: `http://<board-ip>:9090`
2. Upload both files to `/home/xilinx/`

---

## Part 19: Test in Python (Basic)

### Step 19.1: Load Overlay in Python

SSH to board or open Jupyter notebook:

```python
import pynq
import numpy as np

# Load your custom overlay
ol = pynq.Overlay("/home/xilinx/rfsoc_rhino_fft_wrapper.bit")

# Check what IP blocks are accessible
print(dir(ol))
```

**Expected output:**
```
['usp_rf_data_converter_0', 'xfft_0', 'axi_dma_0', ...]
```

---

### Step 19.2: Test RF Data Converter

```python
# Access RF Data Converter
rfdc = ol.usp_rf_data_converter_0

# Check register map
print(rfdc.register_map)

# Check sampling rate
print(f"ADC Tile 224 Sample Rate: {rfdc.adc_tiles[2].blocks[0].SamplingFreq} MHz")

# Should show: 3200 MHz (3.2 GHz)
# With decimation 16x → 200 MSPS output
```

---

### Step 19.3: Test DMA

```python
# Access DMA
dma = ol.axi_dma_0

# Allocate buffer for FFT output
buffer = pynq.allocate(shape=(8192,), dtype=np.complex64)

# Start DMA transfer
dma.recvchannel.transfer(buffer)

# Wait for completion
dma.recvchannel.wait()

# Check data
print(f"Buffer received: {len(buffer)} samples")
print(f"First 10 values: {buffer[:10]}")
```

**Expected:** Buffer should contain FFT output (complex values)

---

## Part 20: Troubleshooting Common Issues

### Issue 1: Clock Wizard Won't Connect

**Symptom:** Can't connect `clk_adc0` to `clk_wiz_0`

**Solution:** Use external port method:
```
1. Make both ports external
2. Rename both to same name
3. Make internal
```

---

### Issue 2: Synthesis Fails - Timing Not Met

**Symptom:** Critical warning about timing closure

**Solution:**
1. Open **Implemented Design**
2. **Reports → Timing → Report Timing Summary**
3. If WNS (Worst Negative Slack) < -1.0 ns:
   - Reduce FFT target clock from 250 to 200 MHz
   - Add pipeline registers
   - Reduce Clock Wizard output from 250 to 200 MHz

---

### Issue 3: DMA Doesn't Transfer Data

**Symptom:** `dma.wait()` hangs forever

**Possible causes:**
1. Interrupt not connected → Check `s2mm_introut` → `pl_ps_irq0`
2. Clocks wrong → Verify all clocks connected
3. Reset stuck → Check reset block connections

---

### Issue 4: FFT Outputs Garbage

**Symptom:** FFT produces nonsense values

**Possible causes:**
1. RF Data Converter not configured → Check decimation = 16x
2. FFT not clocked correctly → Check Clock Wizard connection
3. Data width mismatch → Verify FFT input width = 16 bits

---

### Issue 5: Python Can't Find IP Blocks

**Symptom:** `AttributeError: 'Overlay' object has no attribute 'xfft_0'`

**Solution:**
1. Check both `.bit` and `.hwh` files are in same directory
2. Reload PYNQ:
```python
import pynq
pynq.Device.active_device.reset()
ol = pynq.Overlay("rfsoc_rhino_fft_wrapper.bit")
```

---

## Summary: What You Built

### Hardware Pipeline:

```
Antenna (60-85 MHz)
    ↓
RF Data Converter ADC (3.2 GHz sampling)
    ↓ (Decimation 16x)
ADC Output (200 MSPS) → clk_adc0 (200 MHz)
    ↓                        ↓
    ↓                   Clock Wizard (250 MHz)
    ↓                        ↓
FFT (8192-point)  ←─────────┘
    ↓
DMA
    ↓
DDR Memory ← Python reads here
```

### Key Features:

✅ **200 MSPS** effective sample rate (3.2 GHz / 16)  
✅ **8192-point FFT** → 24.4 kHz frequency resolution  
✅ **RHINO band (60-85 MHz)** covered by ~1025 bins  
✅ **Clock domain crossing** properly handled via Clock Wizard  
✅ **Frequency-locked** processing (Vlad's solution)  
✅ **DMA to memory** for Python access  
✅ **Interrupt-driven** transfers  

---

## Next Steps

### 1. Verify Decimation Works
Write Python script to check output rate is 200 MSPS

### 2. Validate FFT Output
Generate test tone, verify appears at correct frequency bin

### 3. Measure Frequency Resolution
Confirm: 200 MHz / 8192 = 24.4 kHz per bin

### 4. Integrate with daq_unified.py
Modify acquisition script to use custom overlay

### 5. Add PFB (Future)
Once FFT working, add PFB in Python software layer

---

## Files to Save for GitHub

Add these to your repository:

```
vivado_designs/
├── rfsoc_rhino_fft.bd.pdf     (export block diagram as PDF)
├── rfsoc_rhino_fft_wrapper.bit
├── rfsoc_rhino_fft_wrapper.hwh
├── utilization_report.txt
└── timing_report.txt
```

---

**End of Guide**

**Total time estimate:** 3-4 hours (most is synthesis/implementation waiting)

**Result:** Working 200 MSPS FFT spectrometer bitstream for RHINO! 🎉