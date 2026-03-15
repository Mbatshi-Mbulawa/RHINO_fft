# RHINO CW Calibration

**Continuous Wave Absolute Radiometer Calibration for the RHINO 21cm Experiment**  
University of Manchester / Jodrell Bank Observatory  
Author: Mbatshi Jerry Junior Mbulawa  
Supervisors: Dr Phil Bull, Jordan Norris

---

## What This Repository Is

This repository contains the code and documentation for a CW (Continuous Wave) calibration experiment on the **RFSoC 4x2** board, developed as part of the RHINO (Radio Hydrogen Investigation with a New Observatory) project targeting the **21cm hydrogen line (60–85 MHz)**.

The experiment uses the RFSoC's own DAC to inject a known sine wave tone into the ADC and measures the system's frequency response across 50–200 MHz. The result is a **gain curve** — a map of how sensitive the signal chain is at each frequency — which is used to calibrate science observations.

This work supports a paper being prepared for submission to **RASTI (RAS Techniques and Instruments)**:  
> *Jordan Norris, Philip Bull et al. — "Continuous Wave Absolute Radiometer Calibration"*

---

## Hardware Required

| Component | Detail |
|---|---|
| Board | RFSoC 4x2 by Real Digital |
| FPGA | AMD Zynq UltraScale+ ZU48DR (`xczu48dr-ffvg1517-2-e`) |
| Overlay | rfsoc_sam (loaded automatically) |
| RF cable | SMA-to-SMA coax, connecting DAC_A to ADC_A (loopback) |
| Network | Board at `192.168.2.99`, Jupyter at port 9090 |

**Before running any code:** connect the **DAC_A SMA** to the **ADC_A SMA** port on the board using an RF coax cable. This creates the loopback path the experiment relies on.

---

## Repository Structure

```
rhino_cw_calibration/
│
├── rhino_cw_calibration.py       # Main Python script — all measurement functions
├── rhino_cw_calibration.ipynb    # Jupyter notebook — interactive step-by-step guide
├── rhino_cw_experiment_prep.tex  # LaTeX preparation document (compile with pdflatex)
├── README.md                     # This file
│
└── results/                      # Created automatically when you run the experiment
    ├── rhino_cw_results.npz      # All sweep data as numpy arrays
    ├── rhino_cw_results.csv      # Same data in spreadsheet format
    ├── rhino_cw_gain_curve.png   # Main gain curve plot
    ├── m1_switching_speed.png    # Measurement 1 output
    ├── m3_linearity.png          # Measurement 3 output
    └── m4_noise_floor.png        # Measurement 4 output
```

---

## Quick Start

### Option 1: Jupyter Notebook (recommended for first run)

1. Copy the repository to `/home/xilinx/jupyter_notebooks/` on the board:
   ```bash
   scp -r rhino_cw_calibration/ xilinx@192.168.2.99:/home/xilinx/jupyter_notebooks/
   ```

2. Open Jupyter at `http://192.168.2.99:9090` in your browser.

3. Navigate to `rhino_cw_calibration.ipynb` and run cells **one at a time, from top to bottom**.

4. Each cell explains what it does and what to expect before you run it.

### Option 2: Python Script (for automated/unattended runs)

```bash
# SSH into the board
ssh xilinx@192.168.2.99

# Navigate to the notebook directory
cd /home/xilinx/jupyter_notebooks/rhino_cw_calibration/

# Run the full experiment
python3 rhino_cw_calibration.py
```

This runs all four measurements in sequence and saves results automatically.

---

## Detailed Explanation of the Python Script

The script (`rhino_cw_calibration.py`) is built around three independent layers. This architecture was chosen on Jordan Norris's instruction to **make everything as flexible as possible** — the tone generation method can be swapped without touching any measurement or logging code.

### Layer 1: Tone Generator (`set_tone()`)

```python
def set_tone(freq_target_mhz, amplitude=DAC_AMPLITUDE):
```

This is the only function that needs to change when the hardware backend is upgraded. Currently it uses the **rfsoc_sam NCO** (DAC mixer frequency register).

**Why the DDC offset formula is needed:**  
The rfsoc_sam overlay runs the ADC in DDC (Digital Down Conversion) mode, with the ADC NCO permanently fixed at **−1228.8 MHz**. This cannot be changed in Python at runtime (the xrfdc C driver blocks it). So a signal at 75 MHz applied directly to the ADC would be completely filtered out — the DDC window is centred at 1228.8 MHz.

The solution is to shift the DAC tone so that after the DDC mixes it down, it lands at the target frequency:

```
f_DAC = 1228.8 + f_target  (MHz)
```

For example, to observe a tone at 75 MHz in the ADC spectrum, the DAC NCO is set to 1303.8 MHz.

**Future upgrade path:**  
When the custom Vivado bitstream is available, `set_tone()` will be replaced with AXI-Lite register writes to a Xilinx DDS Compiler IP block. Switching speed will drop from ~10–100 ms to ~1 µs. Everything else in the script stays identical.

### Layer 2: Spectrum Capture (`capture_spectrum()`, `find_peak()`)

```python
def capture_spectrum(n_frames=N_FRAMES):
def find_peak(spectrum_dbfs, freqs_mhz, target_mhz, search_window_mhz=2.0):
```

`capture_spectrum()` reads `n_frames` spectral frames from **channel_22** (the confirmed ADC_A SMA channel), averages them, and returns a power spectrum in **dBFS** (decibels relative to full scale).

**Why averaging in linear power, not dBFS:**  
Averaging directly in dBFS would give the wrong answer because dBFS is a logarithmic scale. The correct method is:
1. Convert each frame from dBFS to linear power: `P_linear = 10^(dBFS/10)`
2. Average the linear powers
3. Convert back: `dBFS_avg = 10 * log10(mean_linear)`

**Why `spectrum_type = 'log'`:**  
The rfsoc_sam overlay has two spectrum modes. Despite its name, `'power'` mode zeros out the DC bin and is poorly suited for SNR measurements. `'log'` mode gives clean dBFS values across all bins. This was debugged and confirmed in Session 1 (2026-03-04).

`find_peak()` searches within a ±1 MHz window around the expected tone frequency and returns the maximum power bin. The window accounts for any small frequency offset between the expected and actual tone position.

### Layer 3: Results Logger (`save_results()`)

```python
def save_results(sweep_freqs, peak_powers, peak_freqs, noise_floor, meta):
```

Saves all results in two formats:
- **`.npz`** (numpy archive): efficient binary format, reloadable with `np.load()`
- **`.csv`**: human-readable, openable in Excel or LibreOffice

Both files include experiment metadata (date, configuration parameters) so results are self-documenting.

---

## The Four Measurements

The script runs four measurements in sequence. Each is independent and can be run on its own.

### Measurement 1: NCO Switching Speed

**Function:** `measure_switching_speed(n_trials=20)`

Switches the DAC from 75 MHz to 100 MHz repeatedly and measures how long the system takes to stabilise at the new frequency. "Stable" is defined as three consecutive ADC spectrum frames with the peak within 2 dB of the first good reading.

**Output:** A distribution of switching times in milliseconds. The maximum switching time × 1.5 is recommended as the `DAC_SETTLE_TIME_S` for the sweep.

**Why this matters:** If you start measuring before the DAC has settled, the ADC spectrum shows a transient — the gain curve will have artefacts at every frequency step. This measurement ensures the dwell time is long enough.

**Expected result:** 10–100 ms per switch. If it exceeds 200 ms, the DDS Compiler option (Option B) becomes more attractive.

---

### Measurement 4: Noise Floor Stability

**Function:** `measure_noise_floor(duration_s=60.0, sample_interval_s=5.0)`

Records the mean ADC noise power in the science band (60–85 MHz) with the DAC transmitter disabled, sampled every `sample_interval_s` seconds for `duration_s` seconds total.

**Output:** A time series of noise floor values (dBFS) and a summary of mean and peak-to-peak drift.

**Why this matters:** Absolute calibration requires knowing the noise baseline. If the noise floor drifts significantly over time (e.g. due to board heating), calibration measurements must be interleaved with science observations more frequently.

**Expected result:** Noise floor ≈ −107 dBFS (confirmed in Session 1), stable to within ±0.5 dB over 60 seconds.

---

### Measurement 3: Amplitude Linearity

**Function:** `measure_amplitude_linearity(freq_mhz=75.0, n_steps=10)`

Sweeps the normalised DAC amplitude from 0.1 to 1.0 in `n_steps` equal steps at a fixed frequency (default 75 MHz, centre of the science band). Records the ADC peak power at each amplitude level.

**Output:** A plot of DAC amplitude (dB) vs ADC peak power (dBFS). For a linear system, this should be a straight line with slope = 1.000. A slope significantly different from 1.0 indicates non-linearity (ADC saturation or front-end compression).

**Why this matters:** The gain curve (Measurement 2) is only meaningful if the system is linear — i.e. if the gain does not depend on the signal level. Non-linearity would mean the calibration must be redone for every different signal amplitude.

**Expected result:** Linear slope within 5% of 1.0 for amplitudes below 0.7. Clipping behaviour expected at amplitude = 1.0.

---

### Measurement 2: Gain Curve Sweep (Main Calibration)

**Function:** `run_gain_curve_sweep(settle_time_s=None)`

This is the primary experiment. Sweeps the CW tone from `SWEEP_START_MHZ` to `SWEEP_STOP_MHZ` in steps of `SWEEP_STEP_MHZ`. At each step:

1. Sets the DAC NCO to `1228.8 + f_target` MHz
2. Waits `settle_time_s` for the PLL to re-lock (from Measurement 1)
3. Averages `N_FRAMES` ADC spectra
4. Records the peak bin power and frequency

**Output:** `rhino_cw_results.npz` and `.csv` with all results, plus a two-panel plot:
- Top panel: full sweep (50–200 MHz)
- Bottom panel: science band zoom (60–85 MHz), with the 21cm target (75 MHz) marked

**Why this matters:** This curve is the direct input to the calibration correction. Science observations are divided (in linear power) by this curve to flatten the frequency response of the instrument.

**Expected result:** A smooth curve with ±2–5 dB variation. Large ripples (> 5 dB) indicate cable reflections — check connector torque. The science band (60–85 MHz) should be particularly smooth.

---

## Configuration Reference

All experiment parameters are set as module-level variables in `rhino_cw_calibration.py` and in Cell 3 of the notebook. The key parameters are:

| Variable | Default | Effect |
|---|---|---|
| `SWEEP_START_MHZ` | 50.0 | Start of frequency sweep |
| `SWEEP_STOP_MHZ` | 200.0 | End of frequency sweep |
| `SWEEP_STEP_MHZ` | 1.0 | Frequency step size (use 0.049 for full resolution) |
| `N_FRAMES` | 20 | Spectral frames averaged per step |
| `DAC_AMPLITUDE` | 0.5 | Normalised DAC output level (0–1) |
| `DAC_SETTLE_TIME_S` | 0.1 | Wait time after NCO change (updated by Measurement 1) |
| `DDC_OFFSET_MHZ` | 1228.8 | Fixed — do not change |

---

## Simulation Mode

If you run the script on a PC without the board connected (e.g. for testing), it will automatically detect that `rfsoc_sam` is not available and enter **simulation mode**. In this mode, all ADC readings are replaced with synthetic data. This lets you verify the pipeline logic and plotting code without hardware.

```bash
# On a PC (no board needed)
python3 rhino_cw_calibration.py
# Output: [WARNING] rfsoc_sam not found. Running in SIMULATION mode.
```

---

## Relationship to the RHINO RFSoC Project

This repository is one component of a larger project. The full context is documented in `rhino_project_bible.tex` (in the main RHINO_fft repository). Key relationships:

- **rfsoc_sam overlay:** Used here for initial experiments. The DDC lock at 1228.8 MHz means direct 60–85 MHz sampling requires a custom Vivado bitstream (in progress, blocked on licence).
- **system_overlay (custom bitstream):** Once the Vivado licence tunnel to digdev4 is resolved, a new bitstream with direct sampling and DMA will replace rfsoc_sam. The `set_tone()` function will be updated accordingly.
- **rhino_hw_processing overlay:** A future overlay with hardware FFT + PFB. The measurement framework in this repository is designed to work with it without modification.

---

## Adding This to GitHub

### If creating a new repository

```bash
# On your local machine, create the repo folder
mkdir rhino_cw_calibration
cd rhino_cw_calibration

# Copy all files here
# Then initialise git
git init
git add .
git commit -m "Initial commit: CW calibration experiment for RHINO"

# Create a new repo on GitHub (via browser or gh CLI), then:
git remote add origin https://github.com/Mbatshi-Mbulawa/rhino_cw_calibration.git
git branch -M main
git push -u origin main
```

### If adding to the existing RHINO_fft repository as a subdirectory

```bash
cd RHINO_fft   # your existing repo
mkdir cw_calibration
cp rhino_cw_calibration.py cw_calibration/
cp rhino_cw_calibration.ipynb cw_calibration/
cp rhino_cw_experiment_prep.tex cw_calibration/
cp README.md cw_calibration/

git add cw_calibration/
git commit -m "Add CW calibration experiment (50-200 MHz sweep, rfsoc_sam)"
git push
```

### Recommended .gitignore entries

```
# Results (large binary files — don't commit these)
*.npz
*.npy
results/

# Compiled LaTeX
*.aux
*.log
*.pdf
*.toc
*.out

# Jupyter checkpoints
.ipynb_checkpoints/

# Python cache
__pycache__/
*.pyc
```

---

## Open Questions (To Be Resolved)

These questions must be answered before the RASTI paper can be finalised:

1. **Absolute power reference:** To claim *absolute* calibration, we need to express the injected power in physical units (Watts or Kelvin). This requires either a calibrated attenuator, a known cable insertion loss measurement, or a characterised DAC output power model. *Discuss with Jordan and Phil.*

2. **Quantitative success criterion:** What does a good calibration look like? Is it gain curve flat to ±1 dB? SNR above 20 dB across the full band? *Needs to be defined before Results can be written.*

3. **Switching speed threshold:** After running Measurement 1, share the result with Jordan to decide whether the rfsoc_sam NCO approach is fast enough or whether DDS Compiler hardware is needed.

4. **Phase coherence (deferred):** Jordan indicated phase probably won't matter for total power measurements but may matter for some calibration components. This is explicitly deferred for now.

---

## Contact

| Person | Role | Email |
|---|---|---|
| Mbatshi Jerry Junior Mbulawa | Student / author | University of Manchester |
| Dr Phil Bull | Supervisor | University of Manchester / UWC |
| Jordan Norris | PhD collaborator | jordan.norris@postgrad.manchester.ac.uk |
| Anthony Holloway | JBO IT (licence) | anthony.holloway@manchester.ac.uk |

---

*University of Manchester — RHINO RFSoC Project — 2026*
