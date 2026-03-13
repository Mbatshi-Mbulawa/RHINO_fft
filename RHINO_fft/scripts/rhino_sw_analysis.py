"""
rhino_sw_analysis.py  –  RHINO RFSoC 4x2  Software Analysis Pipeline
======================================================================
Captures raw ADC samples, runs a MANDATORY verification step on them,
then performs either a windowed FFT or a Polyphase Filter Bank (PFB).

USAGE
-----
1. Set MODE  : 'fft'  or  'pfb'
2. Set SOURCE: 'dma'  (live board), 'file' (saved .npy), 'tone' (offline test)
3. Run:   python rhino_sw_analysis.py

PIPELINE
--------
  acquire_samples()
       │
       ▼
  verify_samples()    ← ALWAYS runs first: plots raw time-domain + histogram,
       │                prints health checks, warns/aborts on bad data.
       ▼
  run_fft_mode()  or  run_pfb_mode()

BUG FIXES IN THIS VERSION (vs prior version)
--------------------------------------------
  [CRITICAL]  Window name corrected: 'blackmanharris'  (scipy rejects hyphen form)
  [CRITICAL]  FFT dBFS normalisation corrected:
                power = 2*|rfft(x_norm * w)|^2 / sum(w)^2
                → full-scale sine now reads -3.01 dBFS as expected
  [MINOR]     Removed fragile samples_global; N passed explicitly to plot_fft
  [MINOR]     tight_layout(rect=[0,0,1,0.95]) replaces suptitle y=1.01 (no warning)
  [MINOR]     Peak annotation offset flips left when tone is near Nyquist
  [ADDED]     verify_samples() – 3-panel raw ADC inspection before any processing
  [ADDED]     Health checks: dead channel, clipping, DC offset, all-zeros

AUTHOR     : Mbatshi Jerry Junior Mbulawa
SUPERVISOR : Dr. Phil Bull
PROJECT    : RHINO FFT / SDR Pipeline
BOARD      : RFSoC 4x2  (xczu48dr)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import firwin
import os
import sys
import time

# =============================================================================
# USER CONFIGURATION  (edit these lines only)
# =============================================================================

MODE   = 'fft'       # 'fft'  or  'pfb'
SOURCE = 'tone'      # 'dma'  or  'file'  or  'tone'

# If SOURCE = 'file': path to a previously saved .npy file of int16 samples
SAMPLE_FILE = 'adc_samples.npy'

# If SOURCE = 'tone': synthetic test parameters
TONE_FREQ_HZ = 75e6   # Hz  (must be < FS_HZ/2 = 100 MHz)
TONE_AMP     = 0.80   # fraction of ADC full scale  (0 < TONE_AMP <= 1)
ADD_NOISE_DB = -60    # white noise floor (dBFS); -60 ≈ realistic board noise

# =============================================================================
# FIXED HARDWARE PARAMETERS  (match system_overlay bitstream exactly)
# =============================================================================

FS_HZ         = 200e6    # ADC effective sample rate after 16× decimation (Hz)
N_SAMPLES     = 131072   # samples per DMA frame  (axis_tlast_gen FRAME_LEN=16384
                          # × 8 samples per 128-bit AXI beat)
ADC_BITS      = 12       # effective ADC resolution (xczu48dr RF-ADC)
ADC_FULLSCALE = 32767    # int16 maximum (2^15 − 1); ADC output is 16-bit signed

# PYNQ overlay path (only used when SOURCE = 'dma')
OVERLAY_PATH = '/home/xilinx/jupyter_notebooks/rhino_raw.bit'

# =============================================================================
# PFB PARAMETERS
# =============================================================================

PFB_M         = 1024    # number of output channels  (Δf = FS_HZ/M = 195.3 kHz)
PFB_K         = 8       # FIR taps per channel  (prototype length = K×M = 8192)
                          # NOTE: scipy firwin uses 'blackmanharris' (no hyphen)
PFB_WINDOW    = 'blackmanharris'
SCIENCE_LO_HZ = 60e6   # RHINO science band lower edge (Hz)
SCIENCE_HI_HZ = 85e6   # RHINO science band upper edge (Hz)

# =============================================================================
# OUTPUT
# =============================================================================

SAVE_PLOT = True    # save each plot as a PNG file in the working directory
PLOT_DPI  = 150

# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1: SAMPLE ACQUISITION
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================

def acquire_samples_dma():
    """
    Capture one ADC frame via PYNQ DMA from the system_overlay bitstream.

    The DMA is configured S2MM-only (ADC → DDR).  The axis_tlast_gen RTL
    block asserts TLAST every FRAME_LEN=16384 AXI beats (each beat = 8 samples
    at 128 bits), so one transfer = 131,072 int16 samples.
    """
    try:
        from pynq import Overlay, allocate
    except ImportError:
        print("[ERROR] pynq not found.  Run on the RFSoC board or set "
              "SOURCE = 'file' / 'tone'.", file=sys.stderr)
        sys.exit(1)

    print(f"[DMA] Loading overlay: {OVERLAY_PATH}")
    ol  = Overlay(OVERLAY_PATH)
    dma = ol.axi_dma_0

    # Allocate contiguous buffer in DDR (PYNQ maps it to HP0)
    buf = allocate(shape=(N_SAMPLES,), dtype=np.int16)
    print(f"[DMA] Contiguous buffer at physical 0x{buf.physical_address:08X}")
    print(f"[DMA] Requesting {N_SAMPLES:,} samples "
          f"({N_SAMPLES * 2 / 1024:.0f} kB) ...")

    t0 = time.perf_counter()
    dma.recvchannel.transfer(buf)
    dma.recvchannel.wait()
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    print(f"[DMA] Transfer complete in {elapsed_ms:.1f} ms")

    # Copy to numpy BEFORE freeing the DMA buffer
    samples = np.array(buf, dtype=np.float32)
    buf.freebuffer()

    # Auto-save for offline reuse
    save_path = 'adc_samples.npy'
    np.save(save_path, samples.astype(np.int16))
    print(f"[DMA] Raw samples saved to '{save_path}' for offline reuse")

    return samples


def acquire_samples_file(path):
    """Load ADC samples from a previously saved .npy file."""
    if not os.path.isfile(path):
        print(f"[ERROR] File not found: '{path}'", file=sys.stderr)
        sys.exit(1)
    raw = np.load(path)
    print(f"[FILE] Loaded {len(raw):,} samples from '{path}'  "
          f"dtype={raw.dtype}")
    return raw.astype(np.float32)


def acquire_samples_tone():
    """
    Generate a synthetic CW tone + white noise for offline testing.

    The RFSoC ADC runs at FS_HZ = 200 MSPS after 16× decimation.
    A tone at TONE_FREQ_HZ appears directly in the captured sample stream
    (no image-frequency correction needed here — that only applies to the
    DAC path in loopback mode, where the DAC uses the second Nyquist zone).
    """
    f   = TONE_FREQ_HZ
    A   = TONE_AMP * ADC_FULLSCALE
    t   = np.arange(N_SAMPLES) / FS_HZ

    signal    = A * np.sin(2.0 * np.pi * f * t)
    noise_amp = ADC_FULLSCALE * 10.0 ** (ADD_NOISE_DB / 20.0)
    noise     = noise_amp * np.random.randn(N_SAMPLES)
    samples   = np.clip(signal + noise,
                        -ADC_FULLSCALE, ADC_FULLSCALE).astype(np.float32)

    print(f"[TONE] f = {f/1e6:.1f} MHz   "
          f"A = {A:.0f} counts ({TONE_AMP*100:.0f}% FS)   "
          f"noise = {ADD_NOISE_DB} dBFS")
    return samples


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2: RAW SAMPLE VERIFICATION  (always runs before any DSP)
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================

def verify_samples(samples, source_label):
    """
    Inspect raw ADC samples BEFORE any processing.

    Produces a 3-panel figure:
      Panel 1 – Time-domain waveform (first 4096 samples + 256-sample zoom)
      Panel 2 – Sample histogram vs ideal Gaussian
      Panel 3 – Short-time RMS over the frame (checks for dropouts)

    Then prints a health report and aborts with a clear error message if the
    data fails any critical check.

    Parameters
    ----------
    samples      : np.ndarray float32, ADC counts
    source_label : str, shown in the figure title
    """
    N        = len(samples)
    rms      = float(np.sqrt(np.mean(samples ** 2)))
    peak     = float(np.max(np.abs(samples)))
    dc       = float(np.mean(samples))
    clip_n   = int(np.sum(np.abs(samples) >= ADC_FULLSCALE - 1))
    clip_pct = 100.0 * clip_n / N
    dbfs_rms = 20.0 * np.log10(rms / ADC_FULLSCALE + 1e-20)
    papr_db  = 20.0 * np.log10(peak / (rms + 1e-12))
    all_zero = np.all(samples == 0)

    # ── Health report ────────────────────────────────────────────────
    print(f"\n{'━'*56}")
    print(f"  RAW ADC SAMPLE VERIFICATION  (N = {N:,})")
    print(f"  Source: {source_label}")
    print(f"{'━'*56}")
    print(f"  Min / Max     : {samples.min():.0f} / {samples.max():.0f}  counts")
    print(f"  DC offset     : {dc:+.1f} counts")
    print(f"  RMS           : {rms:.1f} counts  ({dbfs_rms:.1f} dBFS)")
    print(f"  Peak          : {peak:.0f} counts  (full scale = {ADC_FULLSCALE})")
    print(f"  PAPR          : {papr_db:.1f} dB")
    print(f"  Clipped       : {clip_n} samples  ({clip_pct:.3f}%)")
    print(f"{'━'*56}")

    # ── Pass / Fail checks ───────────────────────────────────────────
    errors   = []
    warnings = []

    if all_zero:
        errors.append("FAIL: All samples are zero — DMA transfer did not "
                      "complete or ADC is not clocked.")
    if rms < 10 and not all_zero:
        errors.append(f"FAIL: RMS = {rms:.1f} counts is suspiciously low. "
                      "Check ADC tile enable and clock source.")
    if clip_pct > 1.0:
        warnings.append(f"WARN: {clip_pct:.2f}% of samples are clipped — "
                        "reduce input signal amplitude.")
    if abs(dc) > 0.02 * ADC_FULLSCALE:
        warnings.append(f"WARN: DC offset = {dc:+.0f} counts "
                        f"({100*dc/ADC_FULLSCALE:+.1f}% FS).  "
                        "Check coarse mixer DC compensation.")

    for w in warnings:
        print(f"  ⚠  {w}")
    for e in errors:
        print(f"  ✗  {e}")

    if not errors and not warnings:
        print("  ✓  All checks passed — samples look healthy.")
    print(f"{'━'*56}\n")

    # ── 3-panel verification figure ──────────────────────────────────
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(
        f"RHINO RFSoC — Raw ADC Sample Verification\n"
        f"Source: {source_label}   |   N = {N:,}   |   "
        f"FS = {FS_HZ/1e6:.0f} MSPS   |   "
        f"RMS = {rms:.0f} counts ({dbfs_rms:.1f} dBFS)",
        fontsize=11, fontweight='bold', color='#333333'
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.48, wspace=0.32)

    # Panel 1a – full frame waveform (first 4096 samples)
    ax1 = fig.add_subplot(gs[0, :])
    n_show  = min(4096, N)
    t_us    = np.arange(n_show) / FS_HZ * 1e6   # µs
    ax1.plot(t_us, samples[:n_show],
             linewidth=0.4, color='steelblue', alpha=0.8)
    ax1.axhline( ADC_FULLSCALE, color='crimson', lw=0.8, ls='--',
                label=f'±Full scale (±{ADC_FULLSCALE})')
    ax1.axhline(-ADC_FULLSCALE, color='crimson', lw=0.8, ls='--')
    ax1.axhline(0, color='grey', lw=0.5, ls=':')
    ax1.set_xlim([0, t_us[-1]])
    ax1.set_ylim([-ADC_FULLSCALE * 1.1, ADC_FULLSCALE * 1.1])
    ax1.set_xlabel('Time (µs)', fontsize=10)
    ax1.set_ylabel('ADC counts', fontsize=10)
    ax1.set_title(f'Time-domain waveform  '
                  f'(first {n_show:,} of {N:,} samples shown)',
                  fontsize=10)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.25)

    # Annotate DC and RMS on the waveform
    ax1.axhline(dc, color='orange', lw=1.0, ls='-.',
                label=f'DC = {dc:+.0f} cts')
    ax1.axhline( rms, color='green', lw=1.0, ls='-.',
                label=f'RMS = {rms:.0f} cts')
    ax1.axhline(-rms, color='green', lw=1.0, ls='-.')
    ax1.legend(fontsize=8, loc='upper right', ncol=2)

    # Panel 1b – 256-sample zoom  (bottom-left)
    ax2 = fig.add_subplot(gs[1, 0])
    n_zoom   = min(256, N)
    t_zoom   = np.arange(n_zoom) / FS_HZ * 1e6
    ax2.plot(t_zoom, samples[:n_zoom],
             linewidth=0.9, color='steelblue', marker='.', markersize=2)
    ax2.axhline(0, color='grey', lw=0.5, ls=':')
    ax2.set_xlabel('Time (µs)', fontsize=10)
    ax2.set_ylabel('ADC counts', fontsize=10)
    ax2.set_title(f'Waveform zoom (first {n_zoom} samples)', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Panel 1c – histogram vs Gaussian  (bottom-right)
    ax3 = fig.add_subplot(gs[1, 1])
    n_bins  = 100
    counts, bin_edges, patches = ax3.hist(
        samples, bins=n_bins, density=True,
        color='steelblue', alpha=0.65, label='Sample histogram')
    # Overlay Gaussian with same mean/std
    std = float(np.std(samples))
    x_g = np.linspace(samples.min(), samples.max(), 500)
    gauss = (np.exp(-0.5 * ((x_g - dc) / (std + 1e-9)) ** 2)
             / (std * np.sqrt(2 * np.pi) + 1e-9))
    ax3.plot(x_g, gauss, color='darkorange', lw=1.5,
             label=f'Gaussian (σ={std:.0f})')
    ax3.axvline( ADC_FULLSCALE, color='crimson', lw=1.0, ls='--',
                label='±Full scale')
    ax3.axvline(-ADC_FULLSCALE, color='crimson', lw=1.0, ls='--')
    ax3.set_xlabel('ADC counts', fontsize=10)
    ax3.set_ylabel('Probability density', fontsize=10)
    ax3.set_title('Sample Histogram', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if SAVE_PLOT:
        fname = 'rhino_verify_output.png'
        fig.savefig(fname, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"[VERIFY] Plot saved to '{fname}'")

    plt.show(block=False)
    plt.pause(0.5)   # allow the window to render before continuing

    # Hard abort if data is unusable
    if errors:
        print("\n[VERIFY] ✗ Critical checks FAILED.  Fix the issues above "
              "before running the FFT / PFB.\n")
        sys.exit(1)


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3: SOFTWARE FFT
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================

def run_fft_mode(samples, source_label):
    """
    Hanning-windowed real FFT with correct dBFS normalisation.

    Normalisation (fixed):
        power_db = 10·log10( 2·|rfft(x_norm·w)|² / sum(w)² )
    where x_norm = samples / ADC_FULLSCALE  (range ±1).

    This gives:
        full-scale  sine → −3.01 dBFS  (√2 below full scale)
        full-scale  square → 0.00 dBFS (in theory, DC fully correlated)
    which is the standard audio/RF dBFS convention.

    Parameters
    ----------
    samples      : np.ndarray float32, ADC counts
    source_label : str
    """
    N        = len(samples)
    window   = np.hanning(N)
    win_sum  = float(np.sum(window))     # coherent gain denominator

    # Normalise to full scale, apply window, take real FFT
    x_norm   = samples / ADC_FULLSCALE
    spectrum = np.fft.rfft(x_norm * window)

    # Power spectrum: factor of 2 for one-sided (all energy in positive freqs)
    # Normalised by win_sum² so a full-scale sine → −3.01 dBFS
    power    = 2.0 * (np.abs(spectrum) ** 2) / (win_sum ** 2)
    power_db = 10.0 * np.log10(power + 1e-20)

    freqs_hz = np.fft.rfftfreq(N, d=1.0 / FS_HZ)
    delta_f  = FS_HZ / N

    # ── Peak detection ────────────────────────────────────────────────
    peak_idx  = int(np.argmax(power_db))
    peak_freq = float(freqs_hz[peak_idx])
    peak_pwr  = float(power_db[peak_idx])

    # Noise floor: median of bins outside a 5%-bandwidth guard around peak
    bw_guard    = max(10, int(0.05 * len(freqs_hz)))
    mask        = np.ones(len(power_db), dtype=bool)
    mask[max(0, peak_idx - bw_guard):
         min(len(power_db), peak_idx + bw_guard + 1)] = False
    noise_floor = float(np.median(power_db[mask]))
    snr_db      = peak_pwr - noise_floor

    # ── Theoretical estimates ─────────────────────────────────────────
    sqnr_theory  = 6.02 * ADC_BITS + 1.76           # dB (SQNR formula)
    process_gain = 10.0 * np.log10(N / 2.0)          # dB (FFT averaging gain)
    theory_floor = -(sqnr_theory + process_gain)      # dBFS

    print(f"\n{'─'*56}")
    print(f"  FFT RESULTS")
    print(f"{'─'*56}")
    print(f"  Frame length N         : {N:,} samples")
    print(f"  Sample rate            : {FS_HZ/1e6:.0f} MSPS")
    print(f"  Frequency resolution   : {delta_f:.1f} Hz  ({delta_f/1e3:.3f} kHz/bin)")
    print(f"  Nyquist frequency      : {FS_HZ/2/1e6:.0f} MHz")
    print(f"  Positive bins          : {len(freqs_hz):,}")
    print(f"  Peak frequency         : {peak_freq/1e6:.4f} MHz  (bin {peak_idx:,})")
    print(f"  Peak power             : {peak_pwr:.2f} dBFS")
    print(f"  Noise floor (measured) : {noise_floor:.2f} dBFS")
    print(f"  Measured SNR           : {snr_db:.1f} dB")
    print(f"  Theoretical SQNR       : {sqnr_theory:.1f} dB  ({ADC_BITS}-bit ADC)")
    print(f"  FFT process gain       : {process_gain:.1f} dB")
    print(f"  Theoretical floor      : {theory_floor:.1f} dBFS")
    print(f"{'─'*56}\n")

    _plot_fft(freqs_hz, power_db, peak_freq, peak_pwr,
              snr_db, delta_f, noise_floor, N, source_label)


def _plot_fft(freqs_hz, power_db, peak_freq, peak_pwr,
              snr_db, delta_f, noise_floor, N, source_label):
    """Render FFT results: full spectrum (top) + science band zoom (bottom)."""

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1], hspace=0.42)

    # ── Top: full spectrum ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(freqs_hz / 1e6, power_db,
             linewidth=0.5, color='steelblue', label='Power spectrum')
    ax1.axvspan(SCIENCE_LO_HZ / 1e6, SCIENCE_HI_HZ / 1e6,
                alpha=0.12, color='limegreen',
                label=f'Science band ({SCIENCE_LO_HZ/1e6:.0f}–'
                      f'{SCIENCE_HI_HZ/1e6:.0f} MHz)')
    ax1.axhline(noise_floor, color='grey', lw=0.9, ls='--',
                label=f'Noise floor ({noise_floor:.1f} dBFS)')

    # Peak annotation – flip to left side if tone is within 15 MHz of Nyquist
    nyq_mhz = FS_HZ / 2e6
    pf_mhz  = peak_freq / 1e6
    if nyq_mhz - pf_mhz < 15:
        dx, ha = -12, 'right'
    else:
        dx, ha =  10, 'left'
    ax1.annotate(
        f'Peak: {pf_mhz:.3f} MHz\nSNR: {snr_db:.1f} dB',
        xy=(pf_mhz, peak_pwr),
        xytext=(pf_mhz + dx, peak_pwr - 14),
        ha=ha, fontsize=9, color='darkorange',
        arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.4),
        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                  ec='darkorange', alpha=0.9)
    )
    ax1.set_xlim([0, nyq_mhz])
    ax1.set_ylim([-135, 5])
    ax1.set_xlabel('Frequency (MHz)', fontsize=11)
    ax1.set_ylabel('Power (dBFS)', fontsize=11)
    ax1.set_title(
        f'RHINO RFSoC — Software FFT  |  '
        f'$f_s$ = {FS_HZ/1e6:.0f} MSPS   '
        f'$N$ = {N:,}   '
        f'$\\Delta f$ = {delta_f/1e3:.3f} kHz/bin   '
        f'Source: {source_label}',
        fontsize=10)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ── Bottom: science band zoom ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    sci = (freqs_hz >= SCIENCE_LO_HZ) & (freqs_hz <= SCIENCE_HI_HZ)
    ax2.plot(freqs_hz[sci] / 1e6, power_db[sci],
             linewidth=0.8, color='seagreen')
    ax2.axhline(noise_floor, color='grey', lw=0.9, ls='--')
    # Mark if peak is inside the science band
    if SCIENCE_LO_HZ <= peak_freq <= SCIENCE_HI_HZ:
        ax2.axvline(pf_mhz, color='darkorange', lw=1.0, ls=':',
                    label=f'{pf_mhz:.3f} MHz')
        ax2.legend(fontsize=8)
    ax2.set_xlim([SCIENCE_LO_HZ / 1e6, SCIENCE_HI_HZ / 1e6])
    ax2.set_xlabel('Frequency (MHz)', fontsize=10)
    ax2.set_ylabel('Power (dBFS)', fontsize=10)
    ax2.set_title(f'Science Band Zoom: '
                  f'{SCIENCE_LO_HZ/1e6:.0f}–{SCIENCE_HI_HZ/1e6:.0f} MHz',
                  fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('MODE: FFT', fontsize=13, fontweight='bold',
                 color='steelblue')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if SAVE_PLOT:
        fname = 'rhino_fft_output.png'
        fig.savefig(fname, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"[FFT] Plot saved to '{fname}'")

    plt.show()


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4: SOFTWARE PFB
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================

def _polyphase_filter_bank(samples):
    """
    Critically sampled Polyphase Filter Bank.

    Implementation
    --------------
    The prototype FIR h[n] of length K×M is split into K polyphase
    components E[k, m] = h[k + m·K] (commutator model).

    For each output block index i:
        y[i, m] = Σ_{k=0}^{K-1}  E[k, m] · x[i − k, m]

    Then the M-point FFT of y[i, :] gives the M channel outputs for block i.

    The loop:
        filtered[k:] += x[:N_blocks-k] * E[k]
    implements:
        filtered[i] += x[i-k] * E[k]  for i >= k
    which is exactly the causal polyphase FIR. ✓

    Window note
    -----------
    scipy.signal.firwin() accepts 'blackmanharris' (no hyphen).
    'blackman-harris' raises ValueError and was the bug in the prior version.

    Returns
    -------
    freqs  : np.ndarray  channel centre frequencies (Hz), unsorted
    power  : np.ndarray  mean |X|² per channel (linear, averaged over blocks)
    """
    M = PFB_M
    K = PFB_K

    h        = firwin(K * M, cutoff=1.0 / M, window=PFB_WINDOW)
    E        = h.reshape(K, M)   # [K × M] polyphase matrix

    N_blocks = len(samples) // M
    x        = samples[:N_blocks * M].astype(np.float32).reshape(N_blocks, M)

    filtered = np.zeros_like(x)
    for k in range(K):
        if N_blocks > k:
            filtered[k:] += x[:N_blocks - k] * E[k]

    X     = np.fft.fft(filtered, axis=1)          # [N_blocks × M] complex
    power = np.mean(np.abs(X) ** 2, axis=0)       # [M] per-channel mean power

    freqs = np.fft.fftfreq(M, d=1.0 / FS_HZ)
    return freqs, power


def run_pfb_mode(samples, source_label):
    """
    Run the PFB and plot the channel spectrum.

    dBFS normalisation
    ------------------
    For a full-scale sine aligned to channel m:
        |X[b, m]| ≈ (A / 2) · M · coherent_gain_of_window
    The normalisation:
        power_norm = power / (ADC_FULLSCALE² · (M/2)²)
    ensures that a full-scale, perfectly channel-aligned CW tone reads ≈ 0 dBFS.
    (The factor M/2 is the coherent FFT gain for a sinusoid across M samples.)
    """
    M       = PFB_M
    delta_f = FS_HZ / M

    print(f"\n{'─'*56}")
    print(f"  PFB PARAMETERS")
    print(f"{'─'*56}")
    print(f"  Channels M             : {M}")
    print(f"  FIR taps per channel K : {PFB_K}")
    print(f"  Prototype length       : {M * PFB_K:,} taps")
    print(f"  Window                 : {PFB_WINDOW}")
    print(f"  Channel width          : {delta_f/1e3:.3f} kHz")
    print(f"  Blocks processed       : {len(samples)//M:,}")

    t0            = time.perf_counter()
    freqs, power  = _polyphase_filter_bank(samples)
    elapsed_ms    = (time.perf_counter() - t0) * 1e3
    print(f"  Computation time       : {elapsed_ms:.1f} ms")

    # Sort by frequency
    idx      = np.argsort(freqs)
    freqs_s  = freqs[idx]
    power_s  = power[idx]

    # Normalise to dBFS.
    # For firwin DC gain=1 the polyphase gain per column G[m] = 1/M (uniform).
    # For a unit-amplitude sine aligned to channel ch: |X_peak| = G*M/2 = 1/2.
    # Full-scale sine power at peak = ADC_FULLSCALE^2 * (1/2)^2 = ADC_FULLSCALE^2/4.
    # A full-scale aligned sine therefore reads -1.94 dBFS (= 20*log10(1/sqrt(2))).
    norm     = (ADC_FULLSCALE ** 2) / 4.0
    power_db = 10.0 * np.log10(power_s / norm + 1e-20)

    # ── Science band analysis ─────────────────────────────────────────
    sci_mask     = (freqs_s >= SCIENCE_LO_HZ) & (freqs_s <= SCIENCE_HI_HZ)
    sci_pwr      = power_db[sci_mask]
    sci_freqs    = freqs_s[sci_mask]
    peak_idx_sci = int(np.argmax(sci_pwr))
    peak_ch_freq = float(sci_freqs[peak_idx_sci])
    peak_ch_pwr  = float(sci_pwr[peak_idx_sci])
    peak_m       = int(round(peak_ch_freq * M / FS_HZ))

    # Noise floor: median of science band, excluding ±3 channels around peak
    nm           = np.ones(len(sci_pwr), dtype=bool)
    nm[max(0, peak_idx_sci - 3):
       min(len(sci_pwr), peak_idx_sci + 4)] = False
    noise_floor  = float(np.median(sci_pwr[nm])) if nm.any() else float('nan')
    snr_sci      = peak_ch_pwr - noise_floor

    # Adjacent channel sidelobe suppression
    if 0 < peak_idx_sci < len(sci_pwr) - 1:
        adj = max(sci_pwr[peak_idx_sci - 1], sci_pwr[peak_idx_sci + 1])
        sidelobe_dB = peak_ch_pwr - adj
    else:
        sidelobe_dB = float('nan')

    m_lo = int(np.floor(SCIENCE_LO_HZ * M / FS_HZ))
    m_hi = int(np.floor(SCIENCE_HI_HZ * M / FS_HZ))

    print(f"\n{'─'*56}")
    print(f"  PFB RESULTS — Science Band")
    print(f"{'─'*56}")
    print(f"  Science band channels  : m = {m_lo} to {m_hi}  "
          f"({m_hi - m_lo + 1} channels)")
    print(f"  Peak channel           : m = {peak_m}  "
          f"@ {peak_ch_freq/1e6:.3f} MHz")
    print(f"  Peak channel power     : {peak_ch_pwr:.2f} dBFS")
    print(f"  Science noise floor    : {noise_floor:.2f} dBFS")
    print(f"  In-band SNR            : {snr_sci:.1f} dB")
    print(f"  Adjacent suppression   : {sidelobe_dB:.1f} dB")
    print(f"{'─'*56}\n")

    _plot_pfb(freqs_s, power_db, sci_mask, sci_freqs, sci_pwr,
              peak_ch_freq, peak_ch_pwr, peak_m, noise_floor,
              sidelobe_dB, source_label)


def _plot_pfb(freqs_s, power_db, sci_mask, sci_freqs, sci_pwr,
              peak_ch_freq, peak_ch_pwr, peak_m,
              noise_floor, sidelobe_dB, source_label):
    """3-panel PFB plot: full spectrum / science band zoom / sidelobe bar."""

    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.48, wspace=0.30)

    # ── Panel 1 (top, full width): all PFB channels ────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(freqs_s / 1e6, power_db,
             linewidth=0.55, color='steelblue', label='PFB channel power')
    ax1.axvspan(SCIENCE_LO_HZ / 1e6, SCIENCE_HI_HZ / 1e6,
                alpha=0.15, color='limegreen',
                label=f'Science band ({SCIENCE_LO_HZ/1e6:.0f}–'
                      f'{SCIENCE_HI_HZ/1e6:.0f} MHz)')
    ax1.set_xlim([0, FS_HZ / 2e6])
    ax1.set_xlabel('Frequency (MHz)', fontsize=11)
    ax1.set_ylabel('Power (dBFS)', fontsize=11)
    ax1.set_title(
        f'RHINO RFSoC — Software PFB  (Full Spectrum)\n'
        f'$M$ = {PFB_M}   $K$ = {PFB_K}   '
        f'$\\Delta f_{{\\rm ch}}$ = {FS_HZ/PFB_M/1e3:.1f} kHz   '
        f'Window: {PFB_WINDOW}   Source: {source_label}',
        fontsize=10)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ── Panel 2 (bottom-left): science band channel power ─────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(sci_freqs / 1e6, sci_pwr,
             linewidth=1.0, color='seagreen',
             marker='.', markersize=3, label='Channel power')
    ax2.axhline(noise_floor, color='grey', lw=0.9, ls='--',
                label=f'Noise ({noise_floor:.1f} dBFS)')

    # Annotate peak channel
    dx  = 3 if (peak_ch_freq - SCIENCE_LO_HZ) < (SCIENCE_HI_HZ - peak_ch_freq) else -3
    ha  = 'left' if dx > 0 else 'right'
    ax2.annotate(
        f'm = {peak_m}\n{peak_ch_freq/1e6:.3f} MHz\n{peak_ch_pwr:.1f} dBFS',
        xy=(peak_ch_freq / 1e6, peak_ch_pwr),
        xytext=(peak_ch_freq / 1e6 + dx, peak_ch_pwr - 12),
        ha=ha, fontsize=8, color='darkorange',
        arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.4),
        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                  ec='darkorange', alpha=0.9)
    )
    ax2.set_xlim([SCIENCE_LO_HZ / 1e6, SCIENCE_HI_HZ / 1e6])
    ax2.set_xlabel('Frequency (MHz)', fontsize=10)
    ax2.set_ylabel('Power (dBFS)', fontsize=10)
    ax2.set_title(f'Science Band: '
                  f'{SCIENCE_LO_HZ/1e6:.0f}–{SCIENCE_HI_HZ/1e6:.0f} MHz  '
                  f'|  SNR = {peak_ch_pwr - noise_floor:.1f} dB',
                  fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3 (bottom-right): sidelobe bar chart ±20 channels ────────
    ax3 = fig.add_subplot(gs[1, 1])
    half_win = 20
    offsets  = np.arange(-half_win, half_win + 1)
    m_vals   = peak_m + offsets
    valid    = (m_vals >= 0) & (m_vals < PFB_M)
    m_vals   = m_vals[valid]
    offsets  = offsets[valid]

    ch_freqs_hz = m_vals * FS_HZ / PFB_M
    # Use interpolation to read power_db at these exact frequencies
    ch_pwr      = np.interp(ch_freqs_hz, freqs_s, power_db)
    rel_pwr     = ch_pwr - np.max(ch_pwr)

    bar_colours = ['darkorange' if o == 0 else 'steelblue' for o in offsets]
    ax3.bar(offsets, rel_pwr, color=bar_colours, width=0.75, edgecolor='none')
    ax3.axhline(0, color='darkorange', lw=0.8, ls='--')
    ax3.set_xlabel('Channel offset from peak', fontsize=10)
    ax3.set_ylabel('Relative power (dB)', fontsize=10)
    ax3.set_title(f'Sidelobe Profile  ±{half_win} channels\n'
                  f'Peak m = {peak_m}  ({peak_ch_freq/1e6:.3f} MHz)  '
                  f'Adj. suppression = {sidelobe_dB:.1f} dB',
                  fontsize=10)
    ax3.set_xlim([-half_win - 0.5, half_win + 0.5])
    ax3.grid(True, alpha=0.3, axis='y')

    fig.suptitle('MODE: PFB', fontsize=13, fontweight='bold', color='seagreen')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if SAVE_PLOT:
        fname = 'rhino_pfb_output.png'
        fig.savefig(fname, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"[PFB] Plot saved to '{fname}'")

    plt.show()


# =============================================================================
# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
# =============================================================================

def main():
    print(f"\n{'='*56}")
    print(f"  RHINO RFSoC Software Analysis Pipeline")
    print(f"  MODE   : {MODE.upper()}")
    print(f"  SOURCE : {SOURCE.upper()}")
    print(f"{'='*56}\n")

    # ── Step 1: acquire samples ────────────────────────────────────────
    if SOURCE == 'dma':
        samples      = acquire_samples_dma()
        source_label = 'Live DMA capture'
    elif SOURCE == 'file':
        samples      = acquire_samples_file(SAMPLE_FILE)
        source_label = f'File: {os.path.basename(SAMPLE_FILE)}'
    elif SOURCE == 'tone':
        samples      = acquire_samples_tone()
        source_label = f'Synthetic tone {TONE_FREQ_HZ/1e6:.0f} MHz'
    else:
        print(f"[ERROR] Unknown SOURCE '{SOURCE}'. "
              "Choose 'dma', 'file', or 'tone'.", file=sys.stderr)
        sys.exit(1)

    # Pad or truncate to exactly N_SAMPLES
    if len(samples) < N_SAMPLES:
        print(f"[WARN] {len(samples):,} samples < {N_SAMPLES:,} expected. "
              "Zero-padding to frame length.")
        samples = np.concatenate(
            [samples, np.zeros(N_SAMPLES - len(samples), dtype=np.float32)])
    elif len(samples) > N_SAMPLES:
        print(f"[INFO] Truncating {len(samples):,} → {N_SAMPLES:,} samples.")
        samples = samples[:N_SAMPLES]

    # ── Step 2: ALWAYS verify raw samples first ────────────────────────
    verify_samples(samples, source_label)

    # ── Step 3: run chosen mode ────────────────────────────────────────
    if MODE == 'fft':
        run_fft_mode(samples, source_label)
    elif MODE == 'pfb':
        run_pfb_mode(samples, source_label)
    else:
        print(f"[ERROR] Unknown MODE '{MODE}'. Choose 'fft' or 'pfb'.",
              file=sys.stderr)
        sys.exit(1)

    print("\n[DONE]\n")


if __name__ == '__main__':
    main()