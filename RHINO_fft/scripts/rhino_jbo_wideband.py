#!/usr/bin/env python3
"""
RHINO Wide-Band Spectrum Survey
================================
Plugs into a real antenna via ADC_A SMA.
Produces a full-bandwidth spectrum (0 - 2457 MHz) and waterfall plot
for illustration in Bella's CobraX foregrounds paper.

Author: Mbatshi Jerry Junior Mbulawa
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rfsoc_sam import overlay as sam_overlay

print("[INIT] Loading overlay...")
ol   = sam_overlay.Overlay()
tx   = ol.radio.transmitter
rx   = ol.radio.receiver
sa   = rx.channel_22.spectrum_analyser

# ── Configuration ─────────────────────────────────────────────────────────────
FFT_SIZE      = 8192      # maximum supported by rfsoc_sam
N_FRAMES_AVG  = 20        # frames averaged per spectrum (reduces noise)
N_WATERFALL   = 64        # number of time rows in waterfall
SAVE_PATH     = "/home/xilinx/jupyter_notebooks/"

# ── Step 1: Make sure the DAC transmitter is OFF ──────────────────────────────
# Critical: do not let the on-board DAC pollute the antenna signal
print("[TX] Disabling transmitter...")
for ch_name in ['channel_00', 'channel_20']:
    cfg = getattr(tx, ch_name).frontend.config
    cfg['transmit_enable'] = False
    cfg['amplitude']       = 0.0
    getattr(tx, ch_name).frontend.config = cfg
print("[TX] Transmitter OFF\n")

# ── Step 2: Configure spectrum analyser ───────────────────────────────────────
sa.fft_size      = FFT_SIZE
sa.spectrum_type = 'log'     # clean dBFS values, no zeroed DC bins
sa.dma_enable    = 1
time.sleep(0.3)

fs_hz     = sa.sample_frequency          # 4915200000.0 Hz
fs_mhz    = fs_hz / 1e6
freq_axis = np.arange(FFT_SIZE) * (fs_mhz / FFT_SIZE)   # 0 to ~4915 MHz
print(f"[INFO] Sample rate : {fs_mhz:.1f} MHz")
print(f"[INFO] Nyquist     : {fs_mhz/2:.1f} MHz")
print(f"[INFO] Freq res    : {fs_mhz/FFT_SIZE*1000:.1f} kHz per bin\n")

# ── Step 3: Capture averaged spectrum ─────────────────────────────────────────
print(f"[SPECTRUM] Capturing {N_FRAMES_AVG}-frame average...")
frames = [sa.get_frame() for _ in range(N_FRAMES_AVG)]
spectrum = np.mean(frames, axis=0)
print(f"  Peak: {spectrum.max():.1f} dBFS at "
      f"{freq_axis[np.argmax(spectrum)]:.1f} MHz")
print(f"  Noise floor: {np.median(spectrum):.1f} dBFS\n")

# ── Step 4: Capture waterfall ──────────────────────────────────────────────────
print(f"[WATERFALL] Capturing {N_WATERFALL} rows...")
waterfall = np.zeros((N_WATERFALL, FFT_SIZE))
for i in range(N_WATERFALL):
    frames = [sa.get_frame() for _ in range(5)]
    waterfall[i] = np.mean(frames, axis=0)
    if i % 16 == 0:
        print(f"  Row {i}/{N_WATERFALL}")
print("[WATERFALL] Done\n")

# ── Step 5: Plot ───────────────────────────────────────────────────────────────
print("[PLOT] Generating figures...")

fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.patch.set_facecolor('#0d0d0d')

# -- Spectrum plot --
ax1 = axes[0]
ax1.set_facecolor('#0d0d0d')
ax1.plot(freq_axis, spectrum, color='#00e5ff', linewidth=0.5, alpha=0.9)
ax1.set_xlim(0, fs_mhz / 2)        # 0 to Nyquist (2457 MHz)
ax1.set_xlabel('Frequency (MHz)', color='white', fontsize=11)
ax1.set_ylabel('Power (dBFS)',    color='white', fontsize=11)
ax1.set_title(
    f'Wide-Band Spectrum — Jodrell Bank | RFSoC 4x2 | '
    f'fs = {fs_mhz:.1f} MHz | FFT = {FFT_SIZE} pts | '
    f'{N_FRAMES_AVG}-frame average',
    color='white', fontsize=11)
ax1.tick_params(colors='white')
ax1.spines[:].set_color('#333333')
ax1.grid(True, color='#1e1e1e', linewidth=0.4)

# Mark a few known RFI bands for context
rfi_bands = {
    'FM Radio\n(87.5–108)':  (87.5,  108),
    'DAB\n(174–240)':        (174,   240),
    'GSM/4G\n(~700–960)':   (700,   960),
    'WiFi/LTE\n(~1.4–2.4G)':(1400, 2400),
}
for label, (f_lo, f_hi) in rfi_bands.items():
    ax1.axvspan(f_lo, f_hi, alpha=0.08, color='#ff4444')
    ax1.text((f_lo + f_hi) / 2, spectrum.max() - 5, label,
             color='#ff8888', fontsize=6, ha='center', va='top')

# -- Waterfall plot --
ax2 = axes[1]
ax2.set_facecolor('#0d0d0d')
# Only show 0 to Nyquist (first half of FFT)
half = FFT_SIZE // 2
wf_img = ax2.imshow(
    waterfall[:, :half],
    aspect='auto',
    extent=[0, fs_mhz / 2, N_WATERFALL, 0],
    cmap='inferno',
    vmin=np.percentile(waterfall, 5),
    vmax=np.percentile(waterfall, 99),
)
plt.colorbar(wf_img, ax=ax2, label='Power (dBFS)').ax.yaxis.label.set_color('white')
ax2.set_xlabel('Frequency (MHz)', color='white', fontsize=11)
ax2.set_ylabel('Time (frames)',   color='white', fontsize=11)
ax2.set_title('Waterfall — Jodrell Bank Wide-Band Survey',
              color='white', fontsize=11)
ax2.tick_params(colors='white')

plt.tight_layout()
out = SAVE_PATH + "rhino_jbo_wideband.png"
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close(fig)
print(f"[PLOT] Saved: {out}")
print("\nDone. Copy rhino_jbo_wideband.png to laptop via SCP.")