# =============================================================================
# rhino_cw_waterfall.py
#
# RHINO RFSoC 4x2 — CW Waterfall
# University of Manchester / Jodrell Bank Observatory
# Author: Mbatshi Jerry Junior Mbulawa
# Date:   2025-05-21
#
# WHAT THIS SCRIPT DOES
# ----------------------
# Produces a waterfall plot and stores the data as a numpy array.
#
# Axes:
#   X axis — ADC frequency channels 
#   Y axis — time (each row = one get_frame() call)
#   Colour — IQ power (dBFS)
#
# Frequency axis convention :
#   fmin  = f_lo - (fs/2)   =  1228.8 - 2457.6 = -1228.8 MHz
#   fmax  = f_lo + (fs/2)   =  1228.8 + 2457.6 =  3686.4 MHz
#   freqs = np.linspace(fmin, fmax, nfft)
#
# The DAC sweeps 1.5–3.5 GHz during the acquisition.
# N_FRAMES_PER_STEP frames are captured at each DAC step before moving on.
# The CW tone appears as a bright vertical stripe at its true RF frequency,
# stepping across the x-axis as the DAC NCO changes.
#
# Pseudocode:
#   spectre = []
#   for i in range(n_seconds):
#       s = acquire_spectrum
#       spectre.append(s)
#   Spectra = np.array(spectra)
#   imshow(spectra) / pcolormesh(spectra)
#
# HARDWARE (confirmed on board 2025-05-18)
# -----------------------------------------
#   from rfsoc_sam.overlay import Overlay
#   rx = ol.radio.receiver.channels[3]     — channel_22, ADC_A SMA
#   tx = ol.radio.transmitter.channels[0]  — channel_00, DAC_A SMA
#   sa = rx.spectrum_analyser
#   sa.dma_enable = 1
#   frame = sa.get_frame()                 — float32[2048] dBFS
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

# ── Board detection ───────────────────────────────────────────────────────────
try:
    from rfsoc_sam.overlay import Overlay
    BOARD_CONNECTED = True
    print("[INFO] rfsoc_sam imported. Board mode active.")
except ImportError:
    BOARD_CONNECTED = False
    print("[WARNING] rfsoc_sam not found. Running in SIMULATION mode.")

# =============================================================================
# CONFIGURATION  — edit these values
# =============================================================================

# ── DAC sweep range ───────────────────────────────────────────────────────────
DAC_START_GHZ = 1.5
DAC_STOP_GHZ  = 3.5

# ── Number of get_frame() calls per DAC frequency step ───────────────────────
# The DAC holds at one frequency for this many frames before stepping.
# More frames per step = longer, brighter stripe per step in the waterfall.
N_FRAMES_PER_STEP = 200

# ── Number of DAC steps across 1.5–3.5 GHz ───────────────────────────────────
# Step size = (3.5 - 1.5) GHz / (N_STEPS - 1)
# e.g. N_STEPS=27 => step = 76.9 MHz  (= N_FFT/32 bins)
#      N_STEPS=14 => step = 153.8 MHz (= N_FFT/16 bins)
#      N_STEPS=53 => step = 38.5 MHz  (= N_FFT/64 bins)
N_STEPS = 27

# ── DAC amplitude (1.0 = full scale, confirmed safe on hardware) ──────────────
DAC_AMPLITUDE = 1.0

# ── Settle time after each DAC NCO change ────────────────────────────────────
DAC_SETTLE_TIME_S = 0.5

# ── Output files ─────────────────────────────────────────────────────────────
OUTPUT_NPY      = "rhino_waterfall_spectra.npy"    # raw 2D array (n_rows, nfft)
OUTPUT_FREQS    = "rhino_waterfall_freqs.npy"      # frequency axis (nfft,)
OUTPUT_PLOT     = "rhino_waterfall.png"

# =============================================================================
# HARDWARE CONSTANTS (fixed by rfsoc_sam bitstream — do not change)
# =============================================================================
NFFT   = 2048        # fixed by SpectrumAnalyser IP
F_LO   = 1228.8      # MHz — DDC NCO, fixed by bitstream
FS_MHZ = 4915.2      # MSPS — ADC hardware sample rate

# =============================================================================
# FREQUENCY AXIS 
# =============================================================================
# fmin = f_lo - (fs/2)
# fmax = (fs/2) + f_lo
# freqs = np.linspace(fmin, fmax, nfft)

FMIN_MHZ  = F_LO - (FS_MHZ / 2)       # -1228.8 MHz
FMAX_MHZ  = (FS_MHZ / 2) + F_LO       #  3686.4 MHz
FREQS_MHZ = np.linspace(FMIN_MHZ, FMAX_MHZ, NFFT)

# Total frames in the waterfall
N_TOTAL_FRAMES = N_STEPS * N_FRAMES_PER_STEP

# DAC frequencies for each step
DAC_FREQS_GHZ = np.linspace(DAC_START_GHZ, DAC_STOP_GHZ, N_STEPS)

print(f"[CONFIG] Frequency axis: {FMIN_MHZ:.1f} to {FMAX_MHZ:.1f} MHz")
print(f"[CONFIG] DAC sweep:      {DAC_START_GHZ} to {DAC_STOP_GHZ} GHz  ({N_STEPS} steps)")
print(f"[CONFIG] Frames/step:    {N_FRAMES_PER_STEP}")
print(f"[CONFIG] Total frames:   {N_TOTAL_FRAMES}  ({N_TOTAL_FRAMES * 0.0016:.0f} s approx)")
print(f"[CONFIG] Array size:     {N_TOTAL_FRAMES} x {NFFT}  "
      f"= {N_TOTAL_FRAMES * NFFT * 4 / 1e6:.0f} MB (float32)")

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
    _receiver    = _overlay.radio.receiver.channels[3]
    _transmitter = _overlay.radio.transmitter.channels[0]
    _sa          = _receiver.spectrum_analyser
    _sa.spectrum_type = 'log'
    _sa.dma_enable = 1
    print("[INFO] Receiver  : channels[3] = channel_22 (ADC_A SMA)")
    print("[INFO] Transmitter: channels[0] = channel_00 (DAC_A SMA)")
    print("[INFO] DMA enabled. Ready.")


# =============================================================================
# TONE CONTROL
# =============================================================================

def set_tone(dac_freq_ghz, amplitude=DAC_AMPLITUDE):
    if not BOARD_CONNECTED:
        return
    _transmitter.frontend.controller.centre_frequency = dac_freq_ghz * 1000.0
    _transmitter.frontend.controller.amplitude = amplitude
    _transmitter.frontend.controller.transmit_enable = True
    time.sleep(DAC_SETTLE_TIME_S)


def disable_tone():
    if not BOARD_CONNECTED:
        return
    _transmitter.frontend.controller.transmit_enable = False
    time.sleep(0.05)


# =============================================================================
# WATERFALL ACQUISITION
# =============================================================================

def acquire_waterfall():
    """
    Acquire the waterfall by calling get_frame() continuously.

    Follows pseudocode exactly:
        spectre = []
        for i in range(n_seconds):
            s = acquire_spectrum
            spectre.append(s)
        Spectra = np.array(spectra)

    The DAC steps through DAC_START_GHZ to DAC_STOP_GHZ.
    At each step, get_frame() is called N_FRAMES_PER_STEP times.
    Each call produces one row in the waterfall.

    Returns
    -------
    Spectra   : np.ndarray shape (N_TOTAL_FRAMES, NFFT), dtype float32
                Each row is one raw get_frame() call in dBFS.
                Rows are ordered in time from top to bottom.
    freqs_mhz : np.ndarray shape (NFFT,)
                Frequency axis in MHz using convention:
                fmin = f_lo - fs/2,  fmax = f_lo + fs/2
    dac_step_rows : list of int
                Row index where each DAC step begins (for labelling the plot).
    """
    print("\n" + "="*60)
    print("WATERFALL ACQUISITION")
    print(f"  DAC: {DAC_START_GHZ}–{DAC_STOP_GHZ} GHz  ({N_STEPS} steps)")
    print(f"  {N_FRAMES_PER_STEP} frames per step  →  {N_TOTAL_FRAMES} rows total")
    print(f"  X axis: {FMIN_MHZ:.1f}–{FMAX_MHZ:.1f} MHz ")
    print("="*60)

    # Pre-allocate the full array — : Spectra = np.array(spectra)
    Spectra = np.zeros((N_TOTAL_FRAMES, NFFT), dtype=np.float32)

    dac_step_rows = []   # row index where each DAC step starts
    row = 0
    t0  = time.time()

    for step_idx, f_ghz in enumerate(DAC_FREQS_GHZ):

        # Set DAC frequency
        set_tone(f_ghz, amplitude=DAC_AMPLITUDE)

        # Simulation: inject a tone at the correct RF frequency bin
        if not BOARD_CONNECTED:
            # The DAC tone at f_ghz should appear at f_ghz * 1000 MHz on the
            # frequency axis FREQS_MHZ
            tone_mhz    = f_ghz * 1000.0
            noise_floor = -102.0

        dac_step_rows.append(row)

        # Capture N_FRAMES_PER_STEP frames
        for frame_idx in range(N_FRAMES_PER_STEP):
            if BOARD_CONNECTED:
                frame = np.array(_sa.get_frame(), dtype=np.float32)
            else:
                # Synthetic frame: noise + tone at correct bin
                frame = np.random.normal(noise_floor, 0.9, NFFT).astype(np.float32)
                tone_bin = int(np.argmin(np.abs(FREQS_MHZ - tone_mhz)))
                frame[tone_bin - 1 : tone_bin + 2] = -72.0 + \
                    np.random.normal(0, 0.5, 3).astype(np.float32)

            Spectra[row] = frame
            row += 1

        elapsed = time.time() - t0
        print(f"  Step {step_idx+1:3d}/{N_STEPS}  "
              f"DAC={f_ghz:.3f} GHz  "
              f"rows {dac_step_rows[-1]}–{row-1}  "
              f"elapsed={elapsed:.1f}s")

    disable_tone()

    elapsed_total = time.time() - t0
    print(f"\n  Done. {row} rows in {elapsed_total:.1f} s  "
          f"(ΔT ≈ {elapsed_total/row*1000:.2f} ms/frame)")

    return Spectra, FREQS_MHZ.copy(), dac_step_rows


# =============================================================================
# SAVE DATA
# =============================================================================

def save_data(Spectra, freqs_mhz):
    """
    Store waterfall and frequency axis as numpy arrays.
    Store the data as a numpy array.
    """
    np.save(OUTPUT_NPY,   Spectra)
    np.save(OUTPUT_FREQS, freqs_mhz)

    # Also save a metadata dict
    meta = {
        "dac_start_ghz"   : DAC_START_GHZ,
        "dac_stop_ghz"    : DAC_STOP_GHZ,
        "n_steps"         : N_STEPS,
        "n_frames_per_step": N_FRAMES_PER_STEP,
        "n_total_frames"  : N_TOTAL_FRAMES,
        "nfft"            : NFFT,
        "f_lo_mhz"        : F_LO,
        "fs_mhz"          : FS_MHZ,
        "fmin_mhz"        : FMIN_MHZ,
        "fmax_mhz"        : FMAX_MHZ,
        "dac_amplitude"   : DAC_AMPLITUDE,
        "date"            : time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    np.save("rhino_waterfall_meta.npy", meta)

    print(f"\n  Saved {OUTPUT_NPY}         "
          f"({Spectra.nbytes / 1e6:.0f} MB, shape {Spectra.shape})")
    print(f"  Saved {OUTPUT_FREQS}    (shape {freqs_mhz.shape})")
    print(f"  Saved rhino_waterfall_meta.npy")
    print(f"\n  To reload:")
    print(f"    Spectra   = np.load('{OUTPUT_NPY}')")
    print(f"    freqs_mhz = np.load('{OUTPUT_FREQS}')")


# =============================================================================
# PLOT WATERFALL
# =============================================================================

def plot_waterfall(Spectra, freqs_mhz, dac_step_rows):
    """
    Plot the waterfall using pcolormesh.

    X axis — RF frequency (MHz): f_lo ± fs/2
    Y axis — time (row index, each row = one get_frame() call)
    Colour — IQ power (dBFS)

    The CW tone appears as a bright vertical stripe. As the DAC steps,
    the stripe jumps to a new frequency — visible as a horizontal
    transition in the waterfall.
    """
    n_rows, n_cols = Spectra.shape

    # Row axis — just the integer row index (proportional to time)
    rows = np.arange(n_rows)

    vmin = float(np.nanpercentile(Spectra, 2))
    vmax = float(np.nanpercentile(Spectra, 99.5))

    fig, ax = plt.subplots(figsize=(14, 8))

    # pcolormesh 
    im = ax.pcolormesh(
        freqs_mhz,     # x: RF frequency (MHz)
        rows,          # y: frame index (time)
        Spectra,
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        shading='auto'
    )

    # Mark each DAC step boundary as a horizontal line
    for i, r in enumerate(dac_step_rows):
        ax.axhline(r, color='red', linewidth=0.5, alpha=0.6)
        # Label with DAC frequency on the right
        ax.text(
            freqs_mhz[-1] + 10, r + N_FRAMES_PER_STEP / 2,
            f"{DAC_FREQS_GHZ[i]:.2f}",
            va='center', ha='left', fontsize=6, color='red'
        )

    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('Frame index (time →)', fontsize=12)
    ax.set_title(
        f"RHINO RFSoC 4x2 — CW Waterfall\n"
        f"X = ADC frequency channels  |  Y = time\n"
        f"DAC: {DAC_START_GHZ}–{DAC_STOP_GHZ} GHz  |  "
        f"{N_FRAMES_PER_STEP} frames/step  |  "
        f"{N_STEPS} steps  |  Amp={DAC_AMPLITUDE:.1f}",
        fontsize=10
    )

    # Secondary x-axis label showing GHz
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2_ticks = np.arange(-1000, 4000, 500)
    ax2.set_xticks(ax2_ticks)
    ax2.set_xticklabels([f"{t/1000:.1f}" for t in ax2_ticks], fontsize=8)
    ax2.set_xlabel('Frequency (GHz)', fontsize=10)

    cb = plt.colorbar(im, ax=ax, label='IQ Power (dBFS)', shrink=0.8)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
    print(f"\n  Saved {OUTPUT_PLOT}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("RHINO RFSoC 4x2 — CW Waterfall")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    initialise_hardware()

    # Acquire:
    # spectre = []
    # for i in range(n_seconds):
    #     s = acquire_spectrum
    #     spectre.append(s)
    # Spectra = np.array(spectra)
    Spectra, freqs_mhz, dac_step_rows = acquire_waterfall()

    # Store as numpy array
    save_data(Spectra, freqs_mhz)

    # Plot: imshow(spectra) / pcolormesh(spectra)
    plot_waterfall(Spectra, freqs_mhz, dac_step_rows)

    print("\n" + "=" * 60)
    print("Done.")
    print(f"  {OUTPUT_NPY}   — waterfall data")
    print(f"  {OUTPUT_FREQS} — frequency axis")
    print(f"  {OUTPUT_PLOT}  — waterfall plot")
    print("=" * 60)


if __name__ == "__main__":
    main()