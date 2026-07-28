#!/usr/bin/env python3
"""
knee_frequency_analysis.py
──────────────────────────────────────────────────────────────────────────────
Jordan / RHINO horn antenna — 1/f knee frequency analysis
Load + ZKL-2+ LNA  |  3-hour observation  |  2025-06-19

WHAT THIS SCRIPT DOES
─────────────────────
1. Loads all checkpoint_N*.npy files from a directory.
   Each checkpoint is a cumulative running-mean spectrum saved every
   `save_every` accumulations.  Shape of each file: (n_freq_bins,).

2. Recovers individual time-window snapshots by differencing consecutive
   cumulative means:
       snapshot(N) = [mean(N)×N − mean(N−Δ)×(N−Δ)] / Δ

3. Converts dB → linear power, then computes the fractional deviation
   of each spectrum from the long-run mean:
       δ(t, ν) = P(t, ν) / P̄(ν)

4. Averages δ across all frequency channels to produce a scalar time
   series F(t) that tracks total-power gain fluctuations.

5. Computes the periodogram PSD of F(t).

6. Fits the standard radiometer 1/f noise model:
       PSD(f) = σ_w² · (1 + (f_k / f)^α)
   to extract the knee frequency f_k, white noise floor σ_w², and
   spectral index α.

7. Produces a 4-panel figure and saves it as a PNG.

USAGE
─────
    python knee_frequency_analysis.py /path/to/data/directory/

The directory must contain:
  - checkpoint_N00050.npy, checkpoint_N00100.npy, … (all checkpoints)
  - metadata.json
  - freq_coarse.npy          (optional — used for the mean spectrum panel)
  - final_mean_coarse.npy    (optional — used for the mean spectrum panel)

DEPENDENCIES
────────────
    pip install numpy scipy matplotlib
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit


# ── Constants ────────────────────────────────────────────────────────────────

FS_MHZ      = 4423.680   # ADC sample rate used during acquisition (MHz)
OBS_HOURS   = 3.0        # total observation duration (hours)
N_CAPTURED  = 2266       # total accumulations captured
SAVE_EVERY  = 50         # accumulations between checkpoints


# ── Helper: load checkpoints ─────────────────────────────────────────────────

def load_checkpoints(data_dir: str):
    """Return (nums, cumulative_dict) sorted by checkpoint index."""
    files = [f for f in os.listdir(data_dir) if f.startswith("checkpoint_") and f.endswith(".npy")]
    if not files:
        sys.exit(f"ERROR: no checkpoint_N*.npy files found in {data_dir}")

    nums = sorted([int(f.split("_N")[1].split(".")[0]) for f in files])
    cumulative = {n: np.load(os.path.join(data_dir, f"checkpoint_N{n:05d}.npy"),
                             allow_pickle=False)
                  for n in nums}
    print(f"Loaded {len(nums)} checkpoints  (N{nums[0]:05d} → N{nums[-1]:05d})")
    return nums, cumulative


# ── Helper: reconstruct snapshots ────────────────────────────────────────────

def reconstruct_snapshots(nums, cumulative, save_every=SAVE_EVERY):
    """
    Convert cumulative means → per-window snapshot spectra.
    Returns array of shape (n_windows, n_freq).
    """
    n_freq = cumulative[nums[0]].shape[0]
    snapshots = []
    prev_sum  = np.zeros(n_freq)

    for n in nums:
        current_sum  = cumulative[n] * n
        window_sum   = current_sum - prev_sum
        window_mean  = window_sum / save_every
        snapshots.append(window_mean)
        prev_sum = current_sum

    snapshots = np.array(snapshots)   # (n_windows, n_freq)
    print(f"Snapshot array shape: {snapshots.shape}  "
          f"(range {snapshots.min():.1f} to {snapshots.max():.1f} dB)")
    return snapshots


# ── Helper: build time axis ───────────────────────────────────────────────────

def build_time_axis(n_windows, obs_hours=OBS_HOURS, n_captured=N_CAPTURED,
                    save_every=SAVE_EVERY):
    t_per_sample = obs_hours * 3600 / n_captured   # seconds per accumulation
    t_per_window = t_per_sample * save_every        # seconds per snapshot
    time_hours   = np.arange(n_windows) * t_per_window / 3600
    return time_hours, t_per_window


# ── Helper: 1/f noise model ───────────────────────────────────────────────────

def psd_model(f, sigma_w2, f_k, alpha):
    """Standard radiometer 1/f noise model."""
    return sigma_w2 * (1.0 + (f_k / f) ** alpha)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_analysis(data_dir: str, output_dir: str):

    # ── Load metadata ──────────────────────────────────────────────────────
    meta_path = os.path.join(data_dir, "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print("Metadata:", json.dumps(meta, indent=2))
    else:
        print("WARNING: metadata.json not found — using defaults")

    obs_hours  = meta.get("obs_hours",  OBS_HOURS)
    n_captured = meta.get("n_captured", N_CAPTURED)
    save_every = meta.get("save_every", SAVE_EVERY)
    config     = meta.get("configuration", "Unknown")
    timestamp  = meta.get("obs_timestamp", "Unknown")

    # ── Step 1: Load checkpoints ───────────────────────────────────────────
    nums, cumulative = load_checkpoints(data_dir)

    # ── Step 2: Reconstruct snapshots ─────────────────────────────────────
    snapshots = reconstruct_snapshots(nums, cumulative, save_every)
    n_windows, n_freq = snapshots.shape

    # ── Step 3: Time axis ─────────────────────────────────────────────────
    time_hours, t_per_window = build_time_axis(n_windows, obs_hours,
                                               n_captured, save_every)
    print(f"Time axis: {time_hours[0]:.3f} → {time_hours[-1]:.3f} hours  "
          f"(window = {t_per_window/60:.1f} min)")

    # ── Step 4: dB → linear, fractional deviation, F(t) ──────────────────
    snaps_lin = 10.0 ** (snapshots / 10.0)
    P_mean    = snaps_lin.mean(axis=0)                      # (n_freq,)
    delta     = snaps_lin / P_mean[np.newaxis, :]           # (n_windows, n_freq)
    F_t       = delta.mean(axis=1)                          # (n_windows,)

    print(f"F(t): mean={F_t.mean():.6f}, std={F_t.std():.6f}")

    # ── Step 5: PSD ───────────────────────────────────────────────────────
    freqs_psd, psd = signal.periodogram(F_t, fs=1.0 / t_per_window)
    freqs_psd = freqs_psd[1:]   # drop DC
    psd       = psd[1:]
    freqs_mHz = freqs_psd * 1000

    # ── Step 6: Fit 1/f model ─────────────────────────────────────────────
    sigma_w2_0 = float(np.median(psd[-5:]))
    f_k_0      = float(freqs_psd[len(freqs_psd) // 4])

    try:
        popt, pcov = curve_fit(
            psd_model, freqs_psd, psd,
            p0=[sigma_w2_0, f_k_0, 1.0],
            bounds=([0, freqs_psd.min() * 0.01, 0.1],
                    [np.inf, freqs_psd.max() * 100, 5.0]),
            maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov))
        sigma_w2_fit, f_k_fit, alpha_fit = popt
        fit_ok = True
    except RuntimeError as e:
        print(f"WARNING: curve_fit failed — {e}")
        sigma_w2_fit, f_k_fit, alpha_fit = sigma_w2_0, f_k_0, 1.0
        perr = np.array([np.nan, np.nan, np.nan])
        fit_ok = False

    knee_period_min = 1.0 / f_k_fit / 60.0

    print("\n=== FIT RESULTS ===")
    print(f"  σ_w²   = {sigma_w2_fit:.5f}  ±  {perr[0]:.5f}")
    print(f"  f_knee = {f_k_fit*1000:.3f} mHz  ±  {perr[1]*1000:.3f} mHz")
    print(f"  α      = {alpha_fit:.3f}  ±  {perr[2]:.3f}")
    print(f"  Knee period: {knee_period_min:.1f} min")

    # Dense fit curve for plotting
    f_dense   = np.logspace(np.log10(freqs_psd.min() * 0.5),
                             np.log10(freqs_psd.max() * 1.5), 500)
    psd_dense = psd_model(f_dense, *popt) if fit_ok else np.full(500, np.nan)

    # ── Optional: load freq & final mean for spectrum panel ───────────────
    freq_path = os.path.join(data_dir, "freq_coarse.npy")
    mean_path = os.path.join(data_dir, "final_mean_coarse.npy")
    has_spectrum = os.path.exists(freq_path) and os.path.exists(mean_path)
    if has_spectrum:
        freq_raw   = np.load(freq_path, allow_pickle=False)
        freq_mhz   = freq_raw * FS_MHZ
        final_mean = np.load(mean_path, allow_pickle=False)
    else:
        print("NOTE: freq_coarse.npy / final_mean_coarse.npy not found — "
              "skipping mean spectrum panel")

    # ── Step 7: Figure ────────────────────────────────────────────────────
    PANEL_BG = "#161b22"
    GRID_COL = "#30363d"
    TEXT_COL = "#e6edf3"
    ACC1     = "#58a6ff"   # data points
    ACC2     = "#f78166"   # fit line
    ACC3     = "#3fb950"   # knee marker
    ACC4     = "#d2a8ff"   # spectrum

    def style_ax(ax, title):
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values():
            sp.set_color(GRID_COL)
        ax.tick_params(colors=TEXT_COL, which="both", length=4)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        ax.title.set_color(TEXT_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.6, alpha=0.7)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.38,
                            left=0.09, right=0.97, top=0.91, bottom=0.09)

    # Panel 1 — Mean spectrum
    ax1 = fig.add_subplot(gs[0, 0])
    if has_spectrum:
        ax1.plot(freq_mhz, final_mean, color=ACC4, linewidth=0.7, alpha=0.9)
        ax1.set_xlim(freq_mhz.min(), freq_mhz.max())
    else:
        ax1.text(0.5, 0.5, "freq_coarse.npy\nnot found",
                 ha="center", va="center", color=TEXT_COL, transform=ax1.transAxes)
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Power (dB)")
    style_ax(ax1, "Mean Spectrum (3 h average)")

    # Panel 2 — F(t) time series
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_hours, F_t, color=ACC1, linewidth=1.2,
             marker="o", markersize=3.5, label="F(t)")
    ax2.axhline(1.0, color=GRID_COL, linewidth=1.0, linestyle="--")
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Fractional power")
    ax2.set_xlim(time_hours.min(), time_hours.max())
    style_ax(ax2, "Mean Fractional Deviation F(t)")

    # Panel 3 — PSD (log-log) + fit
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.loglog(freqs_mHz, psd, "o", color=ACC1, markersize=5,
               label="PSD data", zorder=3)
    if fit_ok:
        ax3.loglog(f_dense * 1000, psd_dense, color=ACC2, linewidth=2.0,
                   label="1/f fit", zorder=4)
        ax3.axhline(sigma_w2_fit, color="#e3b341", linewidth=1.2, linestyle=":",
                    label=f"σ_w² = {sigma_w2_fit:.4f}")
        ax3.axvline(f_k_fit * 1000, color=ACC3, linewidth=1.5, linestyle="--",
                    label=f"f_k = {f_k_fit*1000:.3f} mHz")
    ax3.set_xlabel("Temporal frequency (mHz)")
    ax3.set_ylabel("PSD")
    style_ax(ax3, "Power Spectral Density of F(t)")
    ax3.legend(fontsize=8, facecolor="#21262d", edgecolor=GRID_COL,
               labelcolor=TEXT_COL)

    # Panel 4 — Results summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PANEL_BG)
    for sp in ax4.spines.values():
        sp.set_color(GRID_COL)
    ax4.axis("off")

    summary = (
        "RESULTS SUMMARY\n"
        "─────────────────────────────────\n"
        f"Configuration:  {config}\n"
        f"Obs timestamp:  {timestamp}\n"
        f"Obs duration:   {obs_hours:.1f} hours\n"
        f"Accumulations:  {n_captured}\n"
        f"Time windows:   {n_windows}  (every {save_every} accum.)\n"
        f"Window length:  {t_per_window/60:.1f} min\n"
        "\n"
        "1/f NOISE FIT\n"
        "─────────────────────────────────\n"
        f"σ_w²   =  {sigma_w2_fit:.5f}  ±  {perr[0]:.5f}\n"
        f"f_knee = {f_k_fit*1000:.3f} mHz  ±  {perr[1]*1000:.3f} mHz\n"
        f"α      = {alpha_fit:.3f}  ±  {perr[2]:.3f}\n"
        "\n"
        f"Knee period:  {knee_period_min:.1f} min\n"
        f"              ({knee_period_min/60:.3f} hours)\n"
        "\n"
        "INTERPRETATION\n"
        "─────────────────────────────────\n"
        f"1/f dominates below {f_k_fit*1000:.3f} mHz\n"
        f"White noise floor above {f_k_fit*1000:.3f} mHz\n"
        f"α ≈ {alpha_fit:.1f}  →  steep 1/f² regime"
    )

    ax4.text(0.04, 0.97, summary, transform=ax4.transAxes,
             fontsize=8.8, verticalalignment="top", fontfamily="monospace",
             color=TEXT_COL, linespacing=1.6)

    fig.suptitle(
        f"RHINO — Jordan 1/f Knee Frequency Analysis\n"
        f"{config}  |  {obs_hours:.0f}-hour observation  |  {timestamp}",
        color=TEXT_COL, fontsize=13, fontweight="bold", y=0.97,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "rhino_knee_frequency_analysis.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="RHINO 1/f knee frequency analysis from checkpoint .npy files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "data_dir",
        nargs="?",                    # optional — falls back to DEFAULT_DATA_DIR
        default=None,
        help="Directory containing checkpoint_N*.npy, metadata.json, "
             "freq_coarse.npy, final_mean_coarse.npy  "
             "(default: the path hardcoded in DEFAULT_DATA_DIR)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for the PNG figure "
             "(default: a 'results' folder inside data_dir)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # ── Default paths — edit these if your folder moves ───────────────────
    DEFAULT_DATA_DIR = (
        "/Users/user/Downloads/Manny-Masters/Project/Data/LNA_1hour"
    )
    # ──────────────────────────────────────────────────────────────────────

    args = parse_args()

    data_dir   = args.data_dir  if args.data_dir  else DEFAULT_DATA_DIR
    output_dir = args.output    if args.output     else os.path.join(data_dir, "results")

    if not os.path.isdir(data_dir):
        sys.exit(
            f"ERROR: data directory not found:\n  {data_dir}\n"
            "Either update DEFAULT_DATA_DIR in the script or pass the path as an argument."
        )

    run_analysis(data_dir=data_dir, output_dir=output_dir)
