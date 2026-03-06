#!/usr/bin/env python3
"""
RHINO Stepped Tone Verification
================================
Sweeps DAC tone in 10 MHz steps and captures ADC spectrum at each step.
Produces a single figure with one subplot per frequency step.
Full bandwidth display (no zoom) — matches tone verification plot style.

Author: Mbatshi Jerry Junior Mbulawa
Supervisor: Dr. Phil Bull
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rfsoc_sam import overlay as sam_overlay

# =============================================================================
# INITIALISE
# =============================================================================

print("[INIT] Loading overlay...")
ol   = sam_overlay.Overlay()
tx   = ol.radio.transmitter
rfdc = ol.radio.rfdc
sa   = ol.radio.receiver.channel_22.spectrum_analyser
sa.fft_size      = 8192
sa.spectrum_type = 'log'
sa.dma_enable    = 1

fs_hz     = sa.sample_frequency
fs_mhz    = fs_hz / 1e6
freq_axis = np.arange(8192) * (fs_mhz / 8192)
ADC_NCO   = 1228.8

print(f"[INIT] Sample rate : {fs_mhz:.3f} MHz")
print(f"[INIT] Freq res    : {fs_mhz/8192*1e3:.2f} kHz/bin")
print(f"[INIT] ADC NCO     : {ADC_NCO} MHz")
print("[INIT] Done\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

steps_offset_mhz = [110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
dac_freqs        = [ADC_NCO + o for o in steps_offset_mhz]

# =============================================================================
# BASELINE CAPTURE (DAC off)
# =============================================================================

print("[BASELINE] Turning DAC off and capturing noise floor...")
for ch_name in ['channel_00', 'channel_20']:
    cfg = getattr(tx, ch_name).frontend.config
    cfg['transmit_enable'] = False
    cfg['amplitude']       = 0.0
    getattr(tx, ch_name).frontend.config = cfg

time.sleep(0.3)
baseline    = np.mean([sa.get_frame() for _ in range(20)], axis=0)
noise_floor = np.median(baseline)
print(f"[BASELINE] Noise floor: {noise_floor:.1f} dBFS\n")

# =============================================================================
# STEPPED TONE CAPTURE
# =============================================================================

results = []  # (dac_freq, offset, spectrum, peak_freq, peak_db, snr_db)

for dac_freq, offset in zip(dac_freqs, steps_offset_mhz):

    print(f"[STEP] DAC = {dac_freq:.1f} MHz  (expect peak near {offset} MHz)...")

    # Set DAC NCO on both active tiles
    for tile_idx in [0, 2]:
        block        = rfdc.dac_tiles[tile_idx].blocks[0]
        mixer        = block.MixerSettings
        mixer['Freq']        = dac_freq
        mixer['MixerType']   = 2
        mixer['MixerMode']   = 2
        mixer['PhaseOffset'] = 0.0
        block.MixerSettings  = mixer
        block.UpdateEvent(1)

    # Enable transmitter
    for ch_name in ['channel_00', 'channel_20']:
        ch  = getattr(tx, ch_name)
        cfg = ch.frontend.config
        cfg['centre_frequency'] = dac_freq
        cfg['amplitude']        = 0.8
        cfg['transmit_enable']  = True
        ch.frontend.config      = cfg

    time.sleep(0.3)  # Let hardware settle

    # Capture 20-frame average
    spec      = np.mean([sa.get_frame() for _ in range(20)], axis=0)
    peak_bin  = np.argmax(spec)
    peak_freq = freq_axis[peak_bin]
    peak_db   = spec[peak_bin]
    snr_db    = peak_db - noise_floor

    print(f"  Peak: {peak_db:.1f} dBFS at {peak_freq:.2f} MHz  |  SNR: {snr_db:.1f} dB")
    results.append((dac_freq, offset, spec, peak_freq, peak_db, snr_db))

# =============================================================================
# TURN DAC OFF WHEN DONE
# =============================================================================

print("\n[DAC] Disabling transmitter...")
for ch_name in ['channel_00', 'channel_20']:
    cfg = getattr(tx, ch_name).frontend.config
    cfg['transmit_enable'] = False
    cfg['amplitude']       = 0.0
    getattr(tx, ch_name).frontend.config = cfg
print("[DAC] Done\n")

# =============================================================================
# PLOT — full bandwidth, one subplot per step
# =============================================================================

print("[PLOT] Generating stepped tone figure...")

n_steps = len(results)
fig, axes = plt.subplots(n_steps, 1, figsize=(14, 2.8 * n_steps), sharex=True)
fig.patch.set_facecolor('#0d0d0d')

for ax, (dac_freq, offset, spec, peak_freq, peak_db, snr_db) in zip(axes, results):
    ax.set_facecolor('#0d0d0d')

    # Full bandwidth spectrum
    ax.plot(freq_axis, spec, color='#00e5ff', linewidth=0.6, alpha=0.9)

    # Noise floor reference line
    ax.axhline(y=noise_floor, color='#666666', linewidth=0.8,
               linestyle=':', label=f'Noise floor ({noise_floor:.1f} dBFS)')

    # Detected peak marker
    ax.axvline(x=peak_freq, color='#ffff00', linewidth=1.2,
               linestyle='--', alpha=0.9,
               label=f'Peak: {peak_freq:.1f} MHz  |  {peak_db:.1f} dBFS  |  SNR {snr_db:.1f} dB')

    ax.set_ylabel('dBFS', color='white', fontsize=8)
    ax.set_title(
        f'DAC = {dac_freq:.1f} MHz  |  Expected offset = {offset} MHz  |  '
        f'Detected peak = {peak_freq:.1f} MHz  |  SNR = {snr_db:.1f} dB',
        color='white', fontsize=9, pad=4
    )
    ax.tick_params(colors='white', labelsize=7)
    ax.spines[:].set_color('#333333')
    ax.grid(True, color='#1e1e1e', linewidth=0.4)
    ax.set_xlim(freq_axis[0], freq_axis[-1])
    ax.legend(loc='upper right', fontsize=7,
              facecolor='#1a1a1a', edgecolor='#444444', labelcolor='white')

axes[-1].set_xlabel('Frequency (MHz)', color='white', fontsize=11)
fig.suptitle(
    'RHINO RFSoC — Stepped Tone Verification (10 MHz steps, 110–200 MHz offsets)\n'
    'channel_22  |  DDC centre = 1228.8 MHz  |  fs = 4915.2 MHz  |  FFT = 8192 pts',
    color='white', fontsize=12, y=1.005
)

plt.tight_layout()
path = "/home/xilinx/jupyter_notebooks/rhino_stepped_tone.png"
fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
plt.close(fig)
print(f"[PLOT] Saved: {path}")

# =============================================================================
# TERMINAL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("STEPPED TONE SUMMARY")
print("="*60)
print(f"{'DAC Freq':>12}  {'Offset':>8}  {'Peak Freq':>12}  {'SNR':>8}")
print("-"*60)
for dac_freq, offset, _, peak_freq, peak_db, snr_db in results:
    print(f"{dac_freq:>10.1f}  {offset:>6} MHz  "
          f"{peak_freq:>10.2f} MHz  {snr_db:>6.1f} dB")
print("="*60)