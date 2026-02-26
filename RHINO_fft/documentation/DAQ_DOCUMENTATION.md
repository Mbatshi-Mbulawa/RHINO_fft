# RHINO Acquisition System Documentation

**Author:** Mbatshi Jerry Junior Mbulawa  
**Supervisor:** Dr. Bull  
**Date:** February 2025

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Processing Modes](#processing-modes)
4. [Code Structure](#code-structure)
5. [Configuration Guide](#configuration-guide)
6. [Function Reference](#function-reference)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The RHINO acquisition system captures radio frequency spectra from the RFSoC 4x2 board for the 21cm global signal experiment. The system supports two processing modes:

- **FFT Mode**: Uses the hardware FFT built into the `rfsoc_sam` overlay (fast, simple, validated)
- **PFB Mode**: Software-based Polyphase Filter Bank (better frequency resolution, experimental)

**Key Features:**
- Direct RF sampling at 4915.2 MSPS (no decimation)
- Time averaging for noise reduction
- Network transmission to logging server
- Real-time visualization (ASCII + matplotlib)
- Configurable integration time and batch size

---

## System Architecture

```
┌──────────────┐
│   Antenna    │
└──────┬───────┘
       │ RF signal (60-85 MHz)
       ▼
┌──────────────────────────────────────┐
│  RFSoC 4x2 Board                     │
│  ┌────────────────────────────────┐  │
│  │  RF ADC (4915.2 MSPS)          │  │
│  └────────────┬───────────────────┘  │
│               │ Digital samples       │
│               ▼                       │
│  ┌────────────────────────────────┐  │
│  │  FPGA Fabric (rfsoc_sam)       │  │
│  │  • Hardware FFT (8192 pt)      │  │
│  │  • Windowing (Hanning)         │  │
│  │  • Power computation |FFT|²    │  │
│  │  • DMA to DDR memory           │  │
│  └────────────┬───────────────────┘  │
│               │ Spectra               │
│               ▼                       │
│  ┌────────────────────────────────┐  │
│  │  ARM CPU (Python/PYNQ)         │  │
│  │  • Time averaging              │  │
│  │  • PFB processing (optional)   │  │
│  │  • Visualization               │  │
│  │  • Network transmission        │  │
│  └────────────┬───────────────────┘  │
└───────────────┼───────────────────────┘
                │ WebSocket
                ▼
       ┌────────────────┐
       │ Logging Server │
       │ • Data storage │
       │ • MD5 verify   │
       └────────────────┘
```

---

## Processing Modes

### FFT Mode (Recommended)

**How it works:**
1. ADC samples at 4915.2 MSPS continuously
2. FPGA hardware performs 8192-point FFT every frame
3. FPGA averages 32 frames together (hardware averaging)
4. Python reads completed spectrum via DMA (`get_frame()`)
5. Python averages multiple spectra over integration time (software averaging)

**Advantages:**
- Fast (hardware accelerated)
- Proven and validated
- Low CPU usage
- Real-time capable

**Frequency Resolution:**
```
Δf = Sample_Rate / FFT_Size
   = 4915.2 MHz / 8192
   = 600 kHz per bin
```

**RHINO Band Coverage:**
- Band: 60–85 MHz
- Bins: ~42 channels (small fraction of total 8192)

---

### PFB Mode (Experimental)

**How it works:**
1. Raw ADC samples would be captured
2. Software applies polyphase filter decomposition
3. FIR filtering across multiple branches (taps)
4. FFT on filtered data
5. Power spectrum computation

**Advantages:**
- Better frequency resolution (configurable)
- Superior sidelobe suppression
- More flexible (can change parameters)

**Disadvantages:**
- Requires raw ADC sample access (not available in `rfsoc_sam`)
- High CPU usage
- Slower than hardware FFT

**Current Status:**
PFB mode is **not usable** with the `rfsoc_sam` overlay because it only provides FFT output, not raw ADC samples. To use PFB, you would need:
1. Custom Vivado overlay with raw ADC DMA, OR
2. External data file for testing

---

## Code Structure

### File: `daq_unified.py`

```
├── Imports and Setup
│   ├── NumPy, SciPy, time, sys, etc.
│   ├── Matplotlib (optional)
│   └── WebSocket libraries (commented out)
│
├── Configuration (RHINOConfig class)
│   ├── Processing mode selection
│   ├── Network settings
│   ├── FFT parameters
│   ├── PFB parameters
│   └── Timing and control
│
├── Utility Functions
│   ├── db() - Convert to dB scale
│   ├── plot_spectrum_ascii() - Terminal visualization
│   └── plot_spectrum_matplotlib() - Graphical plots
│
├── PolyphaseFilterBank Class
│   ├── __init__() - Initialize PFB
│   ├── _generate_win_coeffs() - Create filter
│   ├── _pfb_fir_frontend() - Polyphase filtering
│   ├── process_block() - Full PFB pipeline
│   └── spectrometer() - Time integration
│
├── RHINOAcquisition Class (MAIN)
│   ├── __init__() - Load overlay, configure hardware
│   ├── _configure_system() - Set FFT/PFB parameters
│   ├── _initialize_pfb() - Setup PFB (if mode=PFB)
│   ├── capture_single_spectrum() - Get one frame (FFT mode)
│   ├── capture_time_averaged_spectrum() - Average multiple frames
│   ├── collect_spectrum_batch() - Collect batch of spectra
│   ├── run_acquisition() - Main loop
│   ├── _transmit_batch() - Send to server (commented out)
│   └── _print_statistics() - Summary at end
│
└── main() - Entry point
    ├── Validation
    ├── Initialization
    └── Run acquisition
```

---

## Configuration Guide

### Essential Settings

**1. Choose Processing Mode:**
```python
PROCESSING_MODE = "FFT"  # Options: "FFT" or "PFB"
```

**2. Enable/Disable Transmission:**
```python
ENABLE_TRANSMISSION = False  # Set True when server is ready
```

**3. Set Integration Time:**
```python
INTEGRATION_TIME_SECONDS = 1.0  # Average spectra over this period
```
- `0.0` = no averaging (single frame per output)
- `1.0` = average ~9000 frames over 1 second
- `10.0` = average ~90,000 frames over 10 seconds

**4. Acquisition Duration:**
```python
ACQUISITION_DURATION = 60  # Total run time in seconds
```

**5. Batch Size:**
```python
SPECTRA_PER_FILE = 10  # How many averaged spectra per batch
```

---

### FFT Mode Settings

```python
FFT_SIZE = 8192              # Points (max for rfsoc_sam)
FFT_WINDOW = 'hanning'       # Window function
FFT_SPECTRUM_TYPE = 'power'  # Output type
NUMBER_OF_FRAMES = 32        # Hardware averaging
```

---

### PFB Mode Settings

```python
PFB_NUM_TAPS = 4          # M: Filter taps (higher = better filter)
PFB_NUM_CHANNELS = 1024   # P: Output channels (FFT size)
PFB_WINDOW_FN = "hamming" # Prototype filter window
```

**Frequency Resolution (PFB):**
```
Δf = Sample_Rate / PFB_NUM_CHANNELS
   = 4915.2 MHz / 1024
   = 4800 kHz per channel
```

---

### Network Settings

```python
SERVER_HOSTNAME = "192.168.1.100"  # IP of logging computer
SERVER_PORT = 8765                  # WebSocket port
```

---

### Visualization Settings

```python
SHOW_ASCII_PLOT = True       # Terminal plot (always works)
SHOW_MATPLOTLIB_PLOT = False # Graphical plot (needs X11 or Jupyter)
```

---

## Function Reference

### Configuration Class

#### `RHINOConfig`

Global configuration parameters. All settings are class variables that can be changed before running.

---

### Utility Functions

#### `db(x)`

Convert linear power values to dB scale.

**Parameters:**
- `x` (ndarray): Linear power values

**Returns:**
- `x_db` (ndarray): Power in dB (10 × log₁₀(x))

**Example:**
```python
linear_power = np.array([100, 1000, 10000])
db_power = db(linear_power)  # [20, 30, 40] dB
```

---

#### `plot_spectrum_ascii(spectrum, freq_axis, n_cols=60, n_rows=10)`

Display spectrum as ASCII bar chart in terminal.

**Parameters:**
- `spectrum` (ndarray): Power spectrum values
- `freq_axis` (ndarray): Frequency axis in MHz
- `n_cols` (int): Width of plot (default 60)
- `n_rows` (int): Height of plot (default 10)

**Output Example:**
```
  -95.3 |████████████████████████████████████████
  -98.7 |████████████████████████████████████████
 -102.1 |██████████████         █████████████████
        +────────────────────────────────────────
         0.0 MHz                      2457.6 MHz
```

---

#### `plot_spectrum_matplotlib(spectrum, freq_axis, title, figsize)`

Create publication-quality matplotlib figure.

**Parameters:**
- `spectrum` (ndarray): Power spectrum, shape `(n_channels,)` or `(n_integrations, n_channels)`
- `freq_axis` (ndarray): Frequency axis in MHz
- `title` (str): Plot title
- `figsize` (tuple): Figure size `(width, height)` in inches

**Returns:**
- `fig`, `ax`: Matplotlib figure and axes objects

**Features:**
- Automatic detection of 1D vs 2D data
- RHINO band (60-85 MHz) highlighted in red
- Grid, labels, and legend

---

### PolyphaseFilterBank Class

#### `__init__(n_taps, n_channels, window_fn)`

Initialize PFB spectrometer.

**Parameters:**
- `n_taps` (int): Number of FIR filter taps (M)
- `n_channels` (int): Number of frequency channels (P)
- `window_fn` (str): Window function ('hamming', 'hanning', 'blackman')

**Reference:** Price (2016), Section 3.3

---

#### `process_block(x)`

Process samples through complete PFB pipeline.

**Parameters:**
- `x` (ndarray): Input time samples, length = W×M×P

**Returns:**
- `x_psd` (ndarray): Power spectral density, shape `(n_spectra, P)`

**Pipeline:**
1. Polyphase decomposition
2. FIR filtering
3. FFT
4. Power computation

---

#### `spectrometer(x, n_integrate)`

Full spectrometer with time integration.

**Parameters:**
- `x` (ndarray): Input samples
- `n_integrate` (int): Number of spectra to average

**Returns:**
- `x_integrated` (ndarray): Time-averaged spectra

---

### RHINOAcquisition Class

#### `__init__(overlay_path=None)`

Initialize acquisition system.

**Parameters:**
- `overlay_path` (str, optional): Path to custom bitstream. If None, loads default `rfsoc_sam` overlay.

**Actions:**
1. Loads FPGA overlay
2. Configures hardware
3. Initializes processing backend (FFT or PFB)
4. Sets up statistics tracking

---

#### `capture_single_spectrum()`

Capture one hardware-averaged spectrum (FFT mode only).

**Returns:**
- `spectrum` (ndarray): Power spectrum, shape `(FFT_SIZE,)`

**Note:** Calls `get_frame()` which blocks until DMA completes.

---

#### `capture_time_averaged_spectrum()`

Capture and average multiple spectra over integration period.

**Returns:**
- `spectrum` (ndarray): Time-averaged power spectrum

**Algorithm:**
1. Calculate number of spectra needed for integration time
2. Accumulate spectra in float64 (precision)
3. Compute mean
4. Convert to float32 (storage efficiency)

---

#### `collect_spectrum_batch(num_averaged_spectra)`

Collect multiple time-averaged spectra into a batch.

**Parameters:**
- `num_averaged_spectra` (int): How many spectra to collect

**Returns:**
- `spectra_batch` (ndarray): Shape `(num_averaged_spectra, n_channels)`
- `batch_duration` (float): Time taken (seconds)

---

#### `run_acquisition()`

Main acquisition loop.

**Algorithm:**
```
While elapsed < ACQUISITION_DURATION:
    1. Collect batch of averaged spectra
    2. Display spectrum (ASCII/matplotlib)
    3. Transmit to server (if enabled)
    4. Save locally (if enabled)
    5. Update statistics
```

**Exit Conditions:**
- Duration reached
- Ctrl+C (keyboard interrupt)
- Fatal error

---

## Usage Examples

### Example 1: Basic FFT Acquisition (Terminal)

```python
# Edit daq_unified.py configuration section:
PROCESSING_MODE = "FFT"
ENABLE_TRANSMISSION = False
INTEGRATION_TIME_SECONDS = 1.0
ACQUISITION_DURATION = 60
SHOW_ASCII_PLOT = True
SHOW_MATPLOTLIB_PLOT = False

# Run on RFSoC board:
$ python3 daq_unified.py
```

**Output:**
- 60 seconds of acquisition
- ASCII spectrum plots in terminal
- Statistics summary at end

---

### Example 2: FFT with Network Transmission

```python
# 1. Start logging server on remote computer:
$ python3 rhino_data_server.py

# 2. Edit daq_unified.py:
ENABLE_TRANSMISSION = True
SERVER_HOSTNAME = "192.168.1.100"  # IP of logging computer

# 3. Uncomment network imports at top of file:
from websockets.sync.client import connect
from pickle import dumps
import itertools

# 4. Run:
$ python3 daq_unified.py
```

---

### Example 3: Quick Test (10 seconds, no transmission)

```python
PROCESSING_MODE = "FFT"
ENABLE_TRANSMISSION = False
INTEGRATION_TIME_SECONDS = 0.0  # No averaging (fast)
ACQUISITION_DURATION = 10
SPECTRA_PER_FILE = 5
```

---

### Example 4: Deep Integration (Noise Reduction)

```python
INTEGRATION_TIME_SECONDS = 10.0  # Average 90,000+ frames
ACQUISITION_DURATION = 600       # 10 minutes
SPECTRA_PER_FILE = 1             # One very deep integration per batch
```

---

### Example 5: Jupyter Notebook Visualization

```python
# In Jupyter cell:
from daq_unified import RHINOAcquisition, RHINOConfig
import matplotlib.pyplot as plt

# Configure
RHINOConfig.PROCESSING_MODE = "FFT"
RHINOConfig.SHOW_MATPLOTLIB_PLOT = True
RHINOConfig.ACQUISITION_DURATION = 30

# Run
acq = RHINOAcquisition()
acq.run_acquisition()

# Plots appear inline in notebook
```

---

## Troubleshooting

### Problem: `get_frame()` hangs forever

**Cause:** DMA not enabled

**Solution:** Verify `self.analyser.dma_enable = 1` in `_configure_system()`

---

### Problem: Spectrum is all zeros or NaN

**Causes:**
1. No antenna connected (expected if testing)
2. ADC not configured correctly
3. Wrong hardware path

**Solution:**
1. Connect antenna (or expect noise floor at -150 to -75 dBFS)
2. Check `self.analyser` path: `ol.radio.receiver.channel_00.spectrum_analyser`
3. Verify overlay loaded: `print(dir(self.ol))`

---

### Problem: Network transmission fails

**Causes:**
1. Server not running
2. Firewall blocking port
3. Wrong IP address

**Solution:**
1. Start server first: `python3 rhino_data_server.py`
2. Check firewall: `sudo ufw allow 8765`
3. Verify IP: `ip addr show`

---

### Problem: FFT size mismatch error

**Cause:** Trying to set FFT_SIZE > 8192

**Solution:** `rfsoc_sam` maximum is 8192. For larger FFT, need custom Vivado overlay.

---

### Problem: PFB mode not working

**Cause:** `rfsoc_sam` doesn't provide raw ADC samples

**Solution:** PFB requires custom overlay. Use FFT mode until Vivado issue resolved.

---

### Problem: "No module named 'rfsoc_sam'"

**Cause:** Library not installed on board

**Solution:**
```bash
# Download zip on laptop, SCP to board, then:
cd /home/xilinx
unzip rfsoc_sam-master.zip
cd rfsoc_sam-master
pip3 install . --break-system-packages
```

---

## References

1. **Price, D.C. (2016).** "Spectrometers and Polyphase Filterbanks in Radio Astronomy." arXiv:1607.03579
2. **Bull et al. (2024).** "RHINO: A large horn antenna for detecting the 21cm global signal." arXiv:2410.00076
3. **University of Strathclyde.** `rfsoc_sam` library. https://github.com/strath-sdr/rfsoc_sam

---

## Appendix: Key Equations

### Frequency Resolution (FFT)
```
Δf = fs / N
```
where `fs` = sample rate, `N` = FFT size

### Frequency Resolution (PFB)
```
Δf = fs / P
```
where `P` = number of PFB channels

### Nyquist Frequency
```
fN = fs / 2
```

### Power Spectrum (dB)
```
P_dB = 10 × log₁₀(|FFT|²)
```

### Processing Gain (Averaging)
```
SNR_improvement = 10 × log₁₀(N_avg)
```
where `N_avg` = number of averaged spectra

---

**End of Documentation**