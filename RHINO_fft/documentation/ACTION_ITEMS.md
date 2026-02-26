# Action Items Based on Supervisor Discussion

**Date:** February 26, 2025  
**Author:** Mbatshi Jerry Junior Mbulawa

---

## Summary of Supervisor Requirements

From the Slack conversation (Feb 25, 9 PM onwards):

### Priority 1: Server/Client and Vivado (This Week)
- Get server/client transmission working
- Apply Vlad's clock fix to Vivado block diagram
- Test FFT output validation

### Priority 2: PFB Implementation
- Implement PFB **without decimation** (supervisor's explicit requirement)
- Full 4915.2 MSPS sample rate
- Focus on algorithm correctness, not optimization

### New Feature Request: DAC Tone Generation
- Generate test tones with DAC
- Frequency comb generation for calibration
- Purpose: Spectral leakage tests, window validation
- Loopback testing: DAC to ADC

### Friday Meeting Discussion Points:
1. Chart course for MSc thesis results
2. Prioritize remaining work
3. Decide: Vivado vs PFB focus

---

## Task 1 Completed: Vivado Clock Fix

### Key Finding: Vlad's response DOES solve the clock problem!

### The Solution (from Vlad):

> "Feed clk_adc0 to the PLL(MMCM) for it to generate s_axi_aclk"

**What this means:**
1. RF Data Converter has TWO clocks:
   - `adc0_clk` (input) — reference clock from Zynq
   - `clk_adc0` (output) — ADC-synchronous clock

2. Use the OUTPUT clock (`clk_adc0`) to drive your processing:
   ```
   RF Data Converter clk_adc0 → Clock Wizard → FFT aclk & DMA aclk
   ```

3. The Clock Wizard (MMCM/PLL) ensures frequency-locking

### What to Do in Vivado:

1. Add Clocking Wizard IP
2. Connect: `clk_adc0` (RF Data Converter output) → Clocking Wizard input
3. Configure: Output clock at 250 MHz
4. Connect: Clock Wizard output → FFT `aclk` + DMA clocks
5. Validate: Should now pass (F6)

See `VIVADO_CLOCK_FIX.md` for detailed instructions.

---

## Task 2 Completed: Python Script Updated

### Changes Made:

#### 1. Network Transmission ENABLED
```python
# Imports uncommented
from websockets.sync.client import connect
from pickle import dumps
import itertools

# Transmission enabled by default
ENABLE_TRANSMISSION = True
```

#### 2. PFB Without Decimation
```python
PFB_USE_DECIMATION = False  # Supervisor requirement
```

#### 3. DAC Tone Generation NEW FEATURE

**New Methods Added:**

##### `generate_dac_tone(freq_mhz, amplitude)`
Generate a single CW tone for testing.

##### `generate_frequency_comb(freq_start, freq_stop, freq_step, amplitude)`
Generate multiple equally-spaced tones (frequency comb).

---

## Files Delivered

1. **`VIVADO_CLOCK_FIX.md`** — Complete Vivado solution
2. **`daq_unified_updated.py`** — Updated script with all features
3. **`ACTION_ITEMS.md`** — This summary

All requirements addressed!