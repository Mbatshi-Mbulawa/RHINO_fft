# =============================================================================
# rhino_cw_iq.py
#
# RHINO RFSoC 4x2 — CW Calibration via Continuous get_frame() Averaging
# University of Manchester / Jodrell Bank Observatory
# Author: Mbatshi Jerry Junior Mbulawa
# Date:   2025-05-21
#
# ARCHITECTURE 
# -----------------------------------------------------------
# For each DAC frequency step:
#   1. Set DAC NCO to target frequency (1.5–3.5 GHz range)
#   2. Call get_frame() continuously N_BLOCKS times in a loop
#      (same pattern as rhino-daq: github.com/RHINO-Experiment/rhino-daq)
#   3. Each frame = 2048 float32 values (N_FFT is FIXED by SpectrumAnalyser IP)
#   4. Average all N_BLOCKS frames in linear power domain
#   5. fftshift to centre DC
#   6. Record peak power and frequency
#   7. Optionally accumulate waterfall (time vs frequency 2D array)
#
# FREQUENCY RANGE 
# --------------------------------------
# DAC NCO sweeps 1.5–3.5 GHz.
# This range is chosen because the RFSoC ZU48DR ADC has no aliasing
# in this region (confirmed hardware passband with no fold-over artefacts).
#
# STEP SIZE
# --------------------------------
# N_FFT = 2048 is FIXED by get_frame(). It cannot be changed.
# Sweep step = N_FFT / STEP_DIVISOR bins.
# STEP_DIVISOR options: 16, 32, 64 (Jordan's specification).
#   1/16  => 128 bins = 153.6 MHz step =>  14 steps across 1.5-3.5 GHz
#   1/32  =>  64 bins =  76.8 MHz step =>  27 steps
#   1/64  =>  32 bins =  38.4 MHz step =>  53 steps
#
# WINDOW
# -----------------------------------
# Rectangular window ONLY — no Hanning, no Blackman.
# Nulls of sinc^2 fall on adjacent channel centres => no leakage
# when CW tone is aligned to a bin centre.
#
# WATERFALL PLOTS
# ----------------
# After the sweep, a waterfall (time vs frequency 2D array) is generated
# showing all N_BLOCKS frames at each frequency step, giving a visual
# picture of how the spectrum evolves over the averaging window.
#
# HARDWARE NOTES (confirmed on board 2025-05-18)
# -----------------------------------------------
# - rfsoc_sam: from rfsoc_sam.overlay import Overlay
# - Receiver:  ol.radio.receiver.channels[3]    (channel_22, ADC_A SMA)
# - Transmitter: ol.radio.transmitter.channels[0] (channel_00, DAC_A SMA)
# - Spectrum:  rx.spectrum_analyser.get_frame()  => float32[2048] dBFS
# - Must set:  sa.dma_enable = 1  before calling get_frame()
# - Freq axis: nu_k = k * fs / (2 * N_FFT), IQ-equiv = 2 * nu_k
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend for board use
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import csv
import os

# ── Board detection ──────────────────────────────────────────────────────────
try:
    from rfsoc_sam.overlay import Overlay
    BOARD_CONNECTED = True
    print("[INFO] rfsoc_sam imported. Board mode active.")
except ImportError:
    BOARD_CONNECTED = False
    print("[WARNING] rfsoc_sam not found. Running in SIMULATION mode.")
    print("          All ADC readings will be synthetic. For testing only.")


# =============================================================================
# CONFIGURATION
# =============================================================================

# ── Frequency range (Jordan: 1.5–3.5 GHz, no aliasing in this band) ─────────
DAC_START_GHZ = 1.5
DAC_STOP_GHZ  = 3.5

# ── Step size: 1/STEP_DIVISOR of total N_FFT channels ────────────────────────
# Jordan: "some division of the total number of channels like 1 in 16, 32, 64"
# 16  => 128 bins = 153.6 MHz step  (fastest, coarsest)
# 32  =>  64 bins =  76.8 MHz step  (medium)
# 64  =>  32 bins =  38.4 MHz step  (finest, slowest)
STEP_DIVISOR = 32          # change to 16 or 64 as needed

# ── N_FFT is FIXED by get_frame() — do not change ────────────────────────────
N_FFT = 2048

# ── Averaging: number of frames to accumulate per step ───────────────────────
# Jordan: "continuously in a similar way to how rhino-daq gets the FFT and averages"
# More blocks = lower noise floor. 100 blocks = 20 dB noise reduction.
N_BLOCKS = 100

# ── DAC amplitude (full scale confirmed safe on hardware 2025-05-18) ─────────
DAC_AMPLITUDE = 1.0

# ── Settle time after DAC NCO change (measured: typical 134 ms, max 1480 ms) ─
DAC_SETTLE_TIME_S = 0.5

# ── Waterfall: how many sweeps to stack for the waterfall plot ────────────────
N_WATERFALL_SWEEPS = 10     # set to 1 to skip waterfall accumulation

# ── ADC hardware sample rate (fixed by rfsoc_sam) ────────────────────────────
ADC_FS_HZ = 4915.2e6

# ── Output files ─────────────────────────────────────────────────────────────
OUTPUT_NPZ       = "rhino_cw_iq_results.npz"
OUTPUT_CSV       = "rhino_cw_iq_results.csv"
OUTPUT_PLOT      = "rhino_cw_iq_gain_curve.png"
OUTPUT_WATERFALL = "rhino_cw_iq_waterfall.png"

# =============================================================================
# DERIVED PARAMETERS (computed from config — do not edit)
# =============================================================================

# Raw bin spacing: fs / (2 * N_FFT)
BIN_SPACING_HZ  = ADC_FS_HZ / (2 * N_FFT)
BIN_SPACING_MHZ = BIN_SPACING_HZ / 1e6

# Step size in bins and MHz
STEP_BINS = N_FFT // STEP_DIVISOR
STEP_MHZ  = STEP_BINS * BIN_SPACING_MHZ

# Number of sweep steps
N_STEPS = int(round((DAC_STOP_GHZ - DAC_START_GHZ) * 1e3 / STEP_MHZ)) + 1

# Frequency axis (raw, 0 to Nyquist) — fftshifted after acquisition
_FREQS_RAW_MHZ = np.linspace(0.0, ADC_FS_HZ / 2e6, N_FFT)

print(f"[CONFIG] N_FFT          = {N_FFT} (fixed by get_frame())")
print(f"[CONFIG] Bin spacing    = {BIN_SPACING_MHZ:.4f} MHz")
print(f"[CONFIG] Step divisor   = 1/{STEP_DIVISOR} => {STEP_BINS} bins = {STEP_MHZ:.1f} MHz")
print(f"[CONFIG] Sweep range    = {DAC_START_GHZ}–{DAC_STOP_GHZ} GHz  ({N_STEPS} steps)")
print(f"[CONFIG] N_BLOCKS       = {N_BLOCKS}  (noise reduction ~{10*np.log10(N_BLOCKS):.1f} dB)")
print(f"[CONFIG] DAC amplitude  = {DAC_AMPLITUDE}")


# =============================================================================
# HARDWARE HANDLES
# =============================================================================
_overlay     = None
_transmitter = None
_receiver    = None
_sa          = None


def initialise_hardware():
    global _overlay, _transmitter, _receiver, _sa

    if not BOARD_CONNECTED:
        print("[SIM] Hardware initialisation skipped.")
        return

    print("[INFO] Loading rfsoc_sam overlay...")
    _overlay     = Overlay()
    _receiver    = _overlay.radio.receiver.channels[3]     # channel_22
    _transmitter = _overlay.radio.transmitter.channels[0]  # channel_00
    _sa          = _receiver.spectrum_analyser

    # Set spectrum type once — log gives dBFS values from get_frame()
    _sa.spectrum_type = 'log'
    _sa.dma_enable = 1      # must be set before any get_frame() call

    print("[INFO] Receiver  : channels[3] = channel_22 (ADC_A SMA)")
    print("[INFO] Transmitter: channels[0] = channel_00 (DAC_A SMA)")
    print("[INFO] DMA enabled. Ready.")


# =============================================================================
# TONE CONTROL
# =============================================================================

def set_tone(dac_freq_ghz, amplitude=DAC_AMPLITUDE):
    """
    Set the DAC NCO to dac_freq_ghz GHz.

    NOTE: This script drives the DAC NCO DIRECTLY in GHz.
    There is no 2x correction formula here because we are targeting
    the DAC frequency directly (not a target IQ spectrum frequency).
    The 1.5-3.5 GHz range falls within the first and second Nyquist
    zones of the ADC where the hardware guarantees no aliasing.
    """
    f_mhz = dac_freq_ghz * 1000.0
    if not BOARD_CONNECTED:
        print(f"[SIM] set_tone: {f_mhz:.1f} MHz  amp={amplitude:.2f}")
        return
    _transmitter.frontend.controller.centre_frequency = f_mhz
    _transmitter.frontend.controller.amplitude = amplitude
    _transmitter.frontend.controller.transmit_enable = True
    time.sleep(DAC_SETTLE_TIME_S)


def disable_tone():
    if not BOARD_CONNECTED:
        print("[SIM] disable_tone.")
        return
    _transmitter.frontend.controller.transmit_enable = False
    time.sleep(0.05)


# =============================================================================
# CORE ACQUISITION — continuous get_frame() loop, then average
# =============================================================================

def acquire_spectrum(n_blocks=N_BLOCKS):
    """
    Acquire an averaged power spectrum by calling get_frame() continuously
    n_blocks times and averaging in the linear power domain.

    This is the pattern from rhino-daq:
      for i in range(n_blocks):
          frame = get_frame()
          accumulate

    Steps:
      1. Call get_frame() n_blocks times (rectangular window — already applied
         inside the SpectrumAnalyser IP)
      2. Average in linear power domain (10^(dBFS/10)) — not dBFS directly,
         to avoid geometric mean bias
      3. Convert back to dBFS
      4. fftshift to centre DC

    Returns
    -------
    avg_dbfs   : np.ndarray[N_FFT]  — averaged power spectrum in dBFS, DC-centred
    freqs_mhz  : np.ndarray[N_FFT]  — frequency axis in MHz, centred at 0
    all_frames : np.ndarray[n_blocks, N_FFT]  — all raw frames (for waterfall)
    """
    if not BOARD_CONNECTED:
        # Simulation: synthetic noise with a tone
        fs = ADC_FS_HZ
        noise_floor = -102.0
        all_frames = np.random.normal(noise_floor, 0.8,
                                      size=(n_blocks, N_FFT)).astype(np.float32)
        # Add a simulated tone at bin 512 (raw freq = 512 * BIN_SPACING_MHZ)
        all_frames[:, 512] = -75.0 + np.random.normal(0, 0.5, n_blocks)
        frames_linear = 10.0 ** (all_frames / 10.0)
        avg_linear    = np.mean(frames_linear, axis=0)
        avg_dbfs      = np.fft.fftshift(10.0 * np.log10(np.maximum(avg_linear, 1e-30)))
        all_frames_shifted = np.fft.fftshift(all_frames, axes=1)
        freqs_mhz = np.fft.fftshift(
            np.fft.fftfreq(N_FFT, d=1.0 / ADC_FS_HZ)) / 1e6
        return avg_dbfs, freqs_mhz, all_frames_shifted

    # Board mode: continuous get_frame() calls
    all_frames = np.zeros((n_blocks, N_FFT), dtype=np.float64)
    for k in range(n_blocks):
        frame = _sa.get_frame()
        all_frames[k] = np.array(frame, dtype=np.float64)

    # Average in linear power domain
    frames_linear = 10.0 ** (all_frames / 10.0)
    avg_linear    = np.mean(frames_linear, axis=0)
    avg_dbfs      = 10.0 * np.log10(np.maximum(avg_linear, 1e-30))

    # fftshift: centre DC
    avg_dbfs_shifted      = np.fft.fftshift(avg_dbfs)
    all_frames_shifted    = np.fft.fftshift(all_frames, axes=1)

    # Frequency axis (centred, in MHz)
    fs_hz     = _sa.sample_frequency
    freqs_mhz = np.fft.fftshift(
        np.fft.fftfreq(N_FFT, d=1.0 / fs_hz)) / 1e6

    return avg_dbfs_shifted, freqs_mhz, all_frames_shifted


def find_peak(spectrum_dbfs, freqs_mhz):
    """Return peak power and its frequency from an fftshifted spectrum."""
    idx = int(np.argmax(spectrum_dbfs))
    return float(spectrum_dbfs[idx]), float(freqs_mhz[idx])


# =============================================================================
# MEASUREMENT 4 — Noise Floor Stability
# =============================================================================

def measure_noise_floor(duration_s=60.0, sample_interval_s=10.0):
    print("\n" + "="*60)
    print("MEASUREMENT 4: IQ Noise Floor Stability")
    print(f"  Duration: {duration_s:.0f} s   Interval: {sample_interval_s:.0f} s")
    print("="*60)

    disable_tone()
    print("  [INFO] Waiting 30 s for board thermal stabilisation...")
    time.sleep(30.0)

    readings = []
    t0 = time.time()

    while time.time() - t0 < duration_s:
        elapsed = time.time() - t0
        spec, freqs, _ = acquire_spectrum()
        nf = float(np.mean(spec))
        readings.append(nf)
        print(f"  t={elapsed:5.1f}s  noise floor: {nf:.2f} dBFS")
        time.sleep(sample_interval_s)

    arr   = np.array(readings)
    mean  = float(np.mean(arr))
    drift = float(np.ptp(arr))
    print(f"\n  Mean: {mean:.2f} dBFS   Drift: {drift:.2f} dB")
    if drift > 1.0:
        print("  [WARNING] Drift > 1 dB — consider longer warmup.")
    else:
        print("  [OK] Noise floor stable.")
    return mean, arr


# =============================================================================
# MEASUREMENT 3 — Amplitude Linearity
# =============================================================================

def measure_amplitude_linearity(dac_freq_ghz=2.5, n_steps=10):
    print("\n" + "="*60)
    print(f"MEASUREMENT 3: Amplitude Linearity  DAC={dac_freq_ghz:.3f} GHz")
    print("="*60)

    amps   = np.linspace(0.1, 1.0, n_steps)
    powers = []

    for amp in amps:
        set_tone(dac_freq_ghz, amplitude=amp)
        spec, freqs, _ = acquire_spectrum()
        pwr, frq = find_peak(spec, freqs)
        powers.append(pwr)
        print(f"  amp={amp:.2f}  peak={pwr:.2f} dBFS  at {frq:.2f} MHz")

    amps   = np.array(amps)
    powers = np.array(powers)
    log_amp = 20.0 * np.log10(amps)
    valid   = ~np.isnan(powers)

    if np.sum(valid) > 2:
        slope, _ = np.polyfit(log_amp[valid], powers[valid], 1)
        print(f"\n  Linearity slope: {slope:.4f}  (ideal = 1.000)")
        print(f"  {'[OK] Linear within 5%.' if abs(slope-1.0) < 0.05 else '[WARNING] Non-linearity detected.'}")

    disable_tone()
    return amps, powers


# =============================================================================
# MEASUREMENT 2 — Gain Curve Sweep + Waterfall
# =============================================================================

def run_gain_curve_sweep(settle_time_s=DAC_SETTLE_TIME_S):
    sweep_freqs_ghz = np.linspace(DAC_START_GHZ, DAC_STOP_GHZ, N_STEPS)

    print("\n" + "="*60)
    print("MEASUREMENT 2: IQ Gain Curve Sweep")
    print(f"  DAC: {DAC_START_GHZ}–{DAC_STOP_GHZ} GHz  "
          f"Step: {STEP_MHZ:.1f} MHz (1/{STEP_DIVISOR} of N_FFT)  "
          f"N_steps: {N_STEPS}")
    print(f"  N_blocks: {N_BLOCKS}  Settle: {settle_time_s:.3f} s")
    print(f"  Rectangular window | fftshift applied")
    print("="*60)

    # Pre-sweep noise floor
    print("\n[1/3] Noise floor (DAC off)...")
    disable_tone()
    nf_spec, nf_freqs, _ = acquire_spectrum(n_blocks=N_BLOCKS * 2)
    noise_floor = float(np.mean(nf_spec))
    print(f"  Noise floor: {noise_floor:.2f} dBFS")

    # Waterfall storage: shape = (N_STEPS * N_BLOCKS, N_FFT)
    # Each row = one raw frame in time order across the full sweep
    waterfall = np.zeros((N_STEPS * N_BLOCKS, N_FFT), dtype=np.float32)
    waterfall_freqs = None   # filled from first step

    # Sweep
    print(f"\n[2/3] Sweeping {N_STEPS} steps...")
    peak_powers = []
    peak_freqs  = []
    t0 = time.time()

    for i, f_ghz in enumerate(sweep_freqs_ghz):
        set_tone(f_ghz, amplitude=DAC_AMPLITUDE)
        time.sleep(settle_time_s)

        spec, freqs, frames = acquire_spectrum(n_blocks=N_BLOCKS)

        if waterfall_freqs is None:
            waterfall_freqs = freqs

        # Store frames in waterfall
        row_start = i * N_BLOCKS
        waterfall[row_start:row_start + N_BLOCKS] = frames.astype(np.float32)

        pwr, frq = find_peak(spec, freqs)
        peak_powers.append(pwr)
        peak_freqs.append(frq)

        snr = pwr - noise_floor if not np.isnan(pwr) else np.nan

        if i % max(1, N_STEPS // 10) == 0 or i == N_STEPS - 1:
            elapsed   = time.time() - t0
            remaining = elapsed / (i + 1) * (N_STEPS - i - 1)
            print(f"  [{i+1:3d}/{N_STEPS}] "
                  f"DAC={f_ghz:.3f} GHz  "
                  f"peak={pwr:7.2f} dBFS  "
                  f"SNR={snr:6.1f} dB  "
                  f"ETA: {remaining:.0f}s")

    disable_tone()
    print(f"\n  Sweep done in {time.time()-t0:.1f} s")

    # Save
    print("\n[3/3] Saving...")
    meta = {
        "date"          : time.strftime("%Y-%m-%d %H:%M:%S"),
        "method"        : "continuous get_frame() average (Jordan Norris 2025-05-19)",
        "window"        : "rectangular (no windowing)",
        "fftshift"      : "applied",
        "dac_start_ghz" : DAC_START_GHZ,
        "dac_stop_ghz"  : DAC_STOP_GHZ,
        "step_divisor"  : STEP_DIVISOR,
        "step_mhz"      : STEP_MHZ,
        "n_steps"       : N_STEPS,
        "n_fft"         : N_FFT,
        "n_blocks"      : N_BLOCKS,
        "dac_amplitude" : DAC_AMPLITUDE,
        "settle_time_s" : settle_time_s,
        "noise_floor"   : noise_floor,
        "bin_spacing_mhz": BIN_SPACING_MHZ,
    }

    np.savez(OUTPUT_NPZ,
             sweep_freqs_ghz = np.array(sweep_freqs_ghz),
             peak_powers     = np.array(peak_powers),
             peak_freqs_mhz  = np.array(peak_freqs),
             noise_floor     = np.array([noise_floor]),
             waterfall       = waterfall,
             waterfall_freqs = np.array(waterfall_freqs) if waterfall_freqs is not None else np.array([]))
    print(f"  Saved {OUTPUT_NPZ}")

    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        for k, v in meta.items():
            w.writerow([f"# {k}", v])
        w.writerow(["dac_freq_ghz", "peak_power_dbfs", "peak_freq_mhz", "snr_db"])
        for sf, pp, pf in zip(sweep_freqs_ghz, peak_powers, peak_freqs):
            snr = pp - noise_floor if not np.isnan(pp) else np.nan
            w.writerow([f"{sf:.4f}", f"{pp:.2f}", f"{pf:.3f}", f"{snr:.2f}"])
    print(f"  Saved {OUTPUT_CSV}")

    return list(sweep_freqs_ghz), peak_powers, peak_freqs, noise_floor, waterfall, waterfall_freqs


# =============================================================================
# PLOTTING — Gain Curve
# =============================================================================

def plot_gain_curve(sweep_freqs_ghz, peak_powers, noise_floor):
    arr   = np.array(sweep_freqs_ghz)
    pwr   = np.array(peak_powers)
    valid = ~np.isnan(pwr)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        f"RHINO RFSoC 4x2 — CW IQ Gain Curve\n"
        f"DAC {DAC_START_GHZ}–{DAC_STOP_GHZ} GHz | "
        f"Rectangular window | fftshift | "
        f"Step=1/{STEP_DIVISOR} ({STEP_MHZ:.0f} MHz) | "
        f"N_blocks={N_BLOCKS} | Amp={DAC_AMPLITUDE:.1f}",
        fontsize=10
    )
    ax.plot(arr[valid], pwr[valid], color='steelblue', lw=1.0,
            label='IQ peak power (dBFS)')
    ax.axhline(noise_floor, color='grey', linestyle='--', lw=0.8,
               label=f'Noise floor ({noise_floor:.1f} dBFS)')
    ax.set_xlabel('DAC NCO frequency (GHz)')
    ax.set_ylabel('IQ Peak Power (dBFS)')
    ax.set_title(f'Gain curve: {DAC_START_GHZ}–{DAC_STOP_GHZ} GHz')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
    print(f"  Saved {OUTPUT_PLOT}")
    plt.close()


# =============================================================================
# PLOTTING — Waterfall
# =============================================================================

def plot_waterfall(waterfall, waterfall_freqs, sweep_freqs_ghz):
    """
    Waterfall plot: rows = time (frame index), columns = frequency.
    Each row is one raw get_frame() call, ordered as the sweep progresses.
    The colour shows IQ power in dBFS.

    This gives a visual picture of:
    - How the tone appears and moves across frequency as the DAC sweeps
    - How stable the noise floor is over time
    - Any RFI or spurious signals that appear at fixed frequencies
    """
    if waterfall is None or waterfall_freqs is None:
        print("  No waterfall data to plot.")
        return

    n_rows, n_cols = waterfall.shape

    # Time axis: each row = one frame; N_BLOCKS frames per step
    time_axis = np.arange(n_rows)
    freq_axis = waterfall_freqs  # MHz, DC-centred

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        f"RHINO RFSoC 4x2 — CW Calibration Waterfall\n"
        f"DAC swept {DAC_START_GHZ}–{DAC_STOP_GHZ} GHz | "
        f"Each row = one get_frame() call | "
        f"{N_BLOCKS} frames per DAC frequency step",
        fontsize=10
    )

    # Use percentile limits for colour scale so bright tones don't wash out noise
    vmin = float(np.percentile(waterfall, 5))
    vmax = float(np.percentile(waterfall, 99))

    im = ax.imshow(
        waterfall,
        aspect='auto',
        origin='lower',
        extent=[freq_axis[0], freq_axis[-1], 0, n_rows],
        vmin=vmin, vmax=vmax,
        cmap='viridis',
        interpolation='nearest'
    )

    # Mark step boundaries
    for step_idx in range(1, N_STEPS):
        ax.axhline(step_idx * N_BLOCKS, color='red', lw=0.5, alpha=0.4)

    # Label y-axis with DAC frequency at each step boundary
    step_rows  = [i * N_BLOCKS + N_BLOCKS // 2 for i in range(N_STEPS)]
    step_labels = [f"{f:.2f}" for f in sweep_freqs_ghz]
    ax.set_yticks(step_rows)
    ax.set_yticklabels(step_labels, fontsize=7)

    ax.set_xlabel('Frequency (MHz, DC-centred)')
    ax.set_ylabel('DAC NCO (GHz)')
    plt.colorbar(im, ax=ax, label='IQ Power (dBFS)', shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_WATERFALL, dpi=150, bbox_inches='tight')
    print(f"  Saved {OUTPUT_WATERFALL}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("RHINO RFSoC 4x2 — CW IQ Calibration")
    print("Jordan Norris architecture — continuous get_frame() averaging")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    initialise_hardware()

    # M4: Noise floor
    noise_mean, noise_arr = measure_noise_floor(
        duration_s=60.0, sample_interval_s=10.0)

    # M3: Linearity at band centre
    mid = (DAC_START_GHZ + DAC_STOP_GHZ) / 2.0
    amps, amp_powers = measure_amplitude_linearity(
        dac_freq_ghz=mid, n_steps=10)

    # M2: Gain curve sweep + waterfall
    sweep_freqs, peak_powers, peak_freqs, nf, waterfall, wf_freqs = \
        run_gain_curve_sweep(settle_time_s=DAC_SETTLE_TIME_S)

    # Plots
    print("\nGenerating plots...")
    plot_gain_curve(sweep_freqs, peak_powers, nf)
    plot_waterfall(waterfall, wf_freqs, sweep_freqs)

    print("\n" + "=" * 60)
    print("Experiment complete.")
    print(f"  {OUTPUT_NPZ}")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_PLOT}")
    print(f"  {OUTPUT_WATERFALL}")
    print("=" * 60)


if __name__ == "__main__":
    main()