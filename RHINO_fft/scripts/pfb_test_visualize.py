#!/usr/bin/env python3
"""
RHINO PFB Spectrometer - Interactive Testing and Visualization
===============================================================

This script is designed for Jupyter notebooks and provides interactive
visualizations of the PFB algorithm.

Can also be run from command line for terminal-based testing.

Author: Mbatshi Jerry Junior Mbulawa
Date: February 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# PFB Parameters
PFB_NUM_TAPS = 4          # M: Number of filter taps
PFB_NUM_CHANNELS = 1024   # P: Number of frequency channels
PFB_WINDOW_FN = "hamming" # Window function

# Simulation parameters
SAMPLE_RATE = 4915.2e6    # RFSoC native rate (Hz)
TEST_FREQ = 75e6          # Test tone frequency (Hz) - in RHINO band
NOISE_LEVEL = 0.1         # Noise standard deviation
SIGNAL_LEVEL = 0.5        # Signal amplitude

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def db(x):
    """Convert linear power to dB."""
    return 10 * np.log10(np.maximum(x, 1e-20))


def generate_test_signal(n_samples, fs, f_signal, noise_std=0.1, signal_amp=0.5):
    """
    Generate test signal: noise + sinusoid.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    fs : float
        Sample rate (Hz)
    f_signal : float
        Signal frequency (Hz)
    noise_std : float
        Noise standard deviation
    signal_amp : float
        Signal amplitude
    
    Returns
    -------
    signal : ndarray
        Generated test signal
    """
    t = np.arange(n_samples)
    noise = np.random.normal(0, noise_std, n_samples)
    omega = 2 * np.pi * f_signal / fs
    tone = signal_amp * np.sin(omega * t)
    return noise + tone


# ══════════════════════════════════════════════════════════════════════════════
# PFB IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_pfb_coefficients(M, P, window_fn="hamming"):
    """Generate prototype filter coefficients for PFB."""
    # Window
    win = scipy.signal.get_window(window_fn, M * P)
    # Sinc with cutoff = 1/P
    sinc = scipy.signal.firwin(M * P, cutoff=1.0/P, window="rectangular")
    # Combine
    h = win * sinc
    # Normalize for processing gain
    pg = np.sum(np.abs(h)**2)
    h /= np.sqrt(pg)
    return h


def pfb_fir_frontend(x, h, M, P):
    """Polyphase FIR filtering stage."""
    W = len(x) // (M * P)
    
    # Polyphase decomposition
    x_p = x.reshape((W * M, P)).T
    h_p = h.reshape((M, P)).T
    
    # Filter and sum
    x_filtered = np.zeros((P, M * W - M + 1))
    for t in range(M * W - M + 1):
        x_slice = x_p[:, t:t + M]
        x_filtered[:, t] = (x_slice * h_p).sum(axis=1)
    
    return x_filtered.T


def pfb_spectrometer(x, M, P, n_integrate=1, window_fn="hamming"):
    """
    Complete PFB spectrometer.
    
    Parameters
    ----------
    x : ndarray
        Input time samples
    M : int
        Number of taps
    P : int
        Number of channels
    n_integrate : int
        Number of spectra to average
    window_fn : str
        Window function name
    
    Returns
    -------
    spectra : ndarray
        Integrated power spectra, shape (n_outputs, P)
    """
    # Trim to valid length
    n_samples = (len(x) // (M * P)) * (M * P)
    x = x[:n_samples]
    
    # Generate filter coefficients
    h = generate_pfb_coefficients(M, P, window_fn)
    
    # FIR frontend
    x_fir = pfb_fir_frontend(x, h, M, P)
    
    # FFT
    x_fft = np.fft.fft(x_fir, n=P, axis=1)
    
    # Power spectrum
    x_psd = np.real(x_fft * np.conj(x_fft))
    
    # Time integration
    n_trim = (x_psd.shape[0] // n_integrate) * n_integrate
    x_psd = x_psd[:n_trim]
    x_psd = x_psd.reshape(-1, n_integrate, P).mean(axis=1)
    
    return x_psd


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_pfb_spectrum(spectrum, freq_axis, title="PFB Spectrum", figsize=(14, 6)):
    """
    Plot PFB output spectrum with annotations.
    
    Parameters
    ----------
    spectrum : ndarray
        Power spectrum, shape (n_channels,) or (n_integrations, n_channels)
    freq_axis : ndarray
        Frequency axis in MHz
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    if spectrum.ndim == 1:
        # Single spectrum - create 2-panel plot
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Panel 1: Full spectrum
        axes[0].plot(freq_axis, db(spectrum), linewidth=1, color='#2E86AB')
        axes[0].set_xlabel('Frequency (MHz)', fontsize=11)
        axes[0].set_ylabel('Power (dB)', fontsize=11)
        axes[0].set_title('Full Spectrum', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].axvspan(60, 85, alpha=0.15, color='red', label='RHINO band')
        axes[0].legend()
        
        # Panel 2: Zoom to RHINO band (60-85 MHz)
        rhino_idx = (freq_axis >= 60) & (freq_axis <= 85)
        if rhino_idx.any():
            axes[1].plot(freq_axis[rhino_idx], db(spectrum[rhino_idx]), 
                        linewidth=1.5, color='#A23B72')
            axes[1].set_xlabel('Frequency (MHz)', fontsize=11)
            axes[1].set_ylabel('Power (dB)', fontsize=11)
            axes[1].set_title('RHINO Band (60-85 MHz)', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3, linestyle='--')
        else:
            axes[1].text(0.5, 0.5, 'RHINO band\nnot in range', 
                        ha='center', va='center', fontsize=14, 
                        transform=axes[1].transAxes)
            axes[1].set_title('RHINO Band (60-85 MHz)', fontsize=12, fontweight='bold')
        
    else:
        # Multiple spectra - waterfall plot
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Panel 1: Waterfall (all frequencies)
        im = axes[0].imshow(db(spectrum), aspect='auto', cmap='viridis',
                           extent=[freq_axis[0], freq_axis[-1], spectrum.shape[0], 0],
                           interpolation='nearest')
        axes[0].set_xlabel('Frequency (MHz)', fontsize=11)
        axes[0].set_ylabel('Integration Number', fontsize=11)
        axes[0].set_title('Spectrogram - Full Band', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[0], label='Power (dB)')
        
        # Panel 2: Average spectrum
        avg_spectrum = spectrum.mean(axis=0)
        axes[1].plot(freq_axis, db(avg_spectrum), linewidth=1, color='#2E86AB')
        axes[1].set_xlabel('Frequency (MHz)', fontsize=11)
        axes[1].set_ylabel('Power (dB)', fontsize=11)
        axes[1].set_title('Time-Averaged Spectrum', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].axvspan(60, 85, alpha=0.15, color='red', label='RHINO band')
        axes[1].legend()
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig, axes


def plot_comparison_fft_vs_pfb(signal, fs, M, P, figsize=(14, 10)):
    """
    Compare standard FFT vs PFB output.
    
    Shows the superior frequency resolution and sidelobe suppression
    of the PFB compared to a simple FFT.
    
    Parameters
    ----------
    signal : ndarray
        Input time signal
    fs : float
        Sample rate (Hz)
    M : int
        PFB taps
    P : int
        PFB channels
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    # Trim signal
    n_valid = (len(signal) // (M * P)) * (M * P)
    signal = signal[:n_valid]
    
    # Standard FFT
    fft_result = np.fft.fft(signal[:P])
    fft_psd = np.real(fft_result * np.conj(fft_result))
    fft_freq = np.fft.fftfreq(P, 1/fs) / 1e6  # MHz
    
    # PFB
    pfb_result = pfb_spectrometer(signal, M, P, n_integrate=1)
    pfb_psd = pfb_result[0]
    pfb_freq = np.arange(P) * (fs / P) / 1e6  # MHz
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Top-left: FFT full spectrum
    axes[0, 0].plot(fft_freq, db(fft_psd), linewidth=1, color='#FF6B35', alpha=0.8)
    axes[0, 0].set_xlabel('Frequency (MHz)', fontsize=11)
    axes[0, 0].set_ylabel('Power (dB)', fontsize=11)
    axes[0, 0].set_title('Standard FFT - Full Spectrum', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvspan(60, 85, alpha=0.15, color='red')
    
    # Top-right: PFB full spectrum
    axes[0, 1].plot(pfb_freq, db(pfb_psd), linewidth=1, color='#2E86AB', alpha=0.8)
    axes[0, 1].set_xlabel('Frequency (MHz)', fontsize=11)
    axes[0, 1].set_ylabel('Power (dB)', fontsize=11)
    axes[0, 1].set_title('PFB - Full Spectrum', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvspan(60, 85, alpha=0.15, color='red')
    
    # Bottom-left: FFT zoomed to RHINO band
    rhino_idx_fft = (fft_freq >= 60) & (fft_freq <= 85)
    if rhino_idx_fft.any():
        axes[1, 0].plot(fft_freq[rhino_idx_fft], db(fft_psd[rhino_idx_fft]), 
                       linewidth=1.5, color='#FF6B35')
        axes[1, 0].set_xlabel('Frequency (MHz)', fontsize=11)
        axes[1, 0].set_ylabel('Power (dB)', fontsize=11)
        axes[1, 0].set_title('FFT - RHINO Band (60-85 MHz)', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Bottom-right: PFB zoomed to RHINO band
    rhino_idx_pfb = (pfb_freq >= 60) & (pfb_freq <= 85)
    if rhino_idx_pfb.any():
        axes[1, 1].plot(pfb_freq[rhino_idx_pfb], db(pfb_psd[rhino_idx_pfb]), 
                       linewidth=1.5, color='#2E86AB')
        axes[1, 1].set_xlabel('Frequency (MHz)', fontsize=11)
        axes[1, 1].set_ylabel('Power (dB)', fontsize=11)
        axes[1, 1].set_title('PFB - RHINO Band (60-85 MHz)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle('Comparison: Standard FFT vs Polyphase Filter Bank', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TEST FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_pfb_test(M=4, P=1024, W=100, n_integrate=10, 
                 fs=SAMPLE_RATE, f_test=TEST_FREQ,
                 show_comparison=True):
    """
    Run complete PFB test with visualizations.
    
    Parameters
    ----------
    M : int
        Number of taps
    P : int
        Number of channels
    W : int
        Number of windows
    n_integrate : int
        Number of spectra to average
    fs : float
        Sample rate (Hz)
    f_test : float
        Test tone frequency (Hz)
    show_comparison : bool
        If True, show FFT vs PFB comparison plot
    
    Returns
    -------
    spectra : ndarray
        PFB output spectra
    """
    print("=" * 70)
    print("RHINO PFB Spectrometer Test")
    print("=" * 70)
    print(f"PFB Parameters:")
    print(f"  Taps (M):              {M}")
    print(f"  Channels (P):          {P}")
    print(f"  Windows (W):           {W}")
    print(f"  Integration:           {n_integrate}")
    print(f"\nSignal Parameters:")
    print(f"  Sample rate:           {fs/1e6:.2f} MHz")
    print(f"  Test tone frequency:   {f_test/1e6:.2f} MHz")
    print(f"  Frequency resolution:  {fs/P/1e3:.3f} kHz per channel")
    print("=" * 70)
    
    # Generate test signal
    n_samples = M * P * W
    print(f"\n[1/4] Generating {n_samples:,} test samples...")
    signal = generate_test_signal(n_samples, fs, f_test, NOISE_LEVEL, SIGNAL_LEVEL)
    
    # Process through PFB
    print(f"[2/4] Processing through PFB...")
    import time
    t_start = time.time()
    spectra = pfb_spectrometer(signal, M, P, n_integrate, PFB_WINDOW_FN)
    t_elapsed = time.time() - t_start
    print(f"      ✓ Complete ({t_elapsed:.3f} s)")
    print(f"      Output shape: {spectra.shape}")
    
    # Find peak
    peak_ch = np.argmax(spectra[0])
    freq_axis = np.arange(P) * (fs / P) / 1e6
    peak_freq = freq_axis[peak_ch]
    print(f"      Peak at channel {peak_ch} ({peak_freq:.2f} MHz)")
    print(f"      Expected: {f_test/1e6:.2f} MHz")
    
    # Create plots
    print(f"[3/4] Creating visualizations...")
    
    # Main spectrum plot
    fig1, _ = plot_pfb_spectrum(spectra[0] if spectra.shape[0] == 1 else spectra, 
                                 freq_axis,
                                 title=f"PFB Output - {f_test/1e6:.1f} MHz Test Tone")
    
    # Comparison plot
    if show_comparison:
        fig2, _ = plot_comparison_fft_vs_pfb(signal, fs, M, P)
    
    print(f"[4/4] Done!")
    print("=" * 70)
    
    # Show plots
    plt.show()
    
    return spectra


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND-LINE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run test
    spectra = run_pfb_test(
        M=PFB_NUM_TAPS,
        P=PFB_NUM_CHANNELS,
        W=100,
        n_integrate=10,
        show_comparison=True
    )
    
    print("\n[INFO] Test complete. Close plot windows to exit.")
    plt.show(block=True)  # Keep plots open until user closes them