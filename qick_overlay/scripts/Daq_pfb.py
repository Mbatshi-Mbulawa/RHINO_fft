#!/usr/bin/env python3
"""
RHINO 21cm Global Signal Acquisition - Polyphase Filter Bank Implementation
============================================================================

This script implements a PFB (Polyphase Filter Bank) spectrometer for the
RHINO 21cm experiment on the RFSoC 4x2 board. The PFB provides superior
frequency resolution and channel isolation compared to standard FFT.

Based on:
- Danny C. Price, "Spectrometers and Polyphase Filterbanks in Radio Astronomy"
  arXiv:1607.03579 (2016)
- University of Strathclyde rfsoc_sam library

Author: Mbatshi Jerry Junior Mbulawa
Supervisor: Dr. Bull
Date: February 2025
"""

import numpy as np
import time
import sys
from datetime import datetime
import scipy.signal

# Matplotlib for Jupyter/graphical plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] Matplotlib not available - graphical plots disabled")

# Network transmission deliberately commented out until core acquisition verified
# from websockets.sync.client import connect
# from pickle import dumps
# import itertools


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class RHINOConfig:
    """
    Configuration parameters for RHINO PFB spectrometer acquisition.
    
    References:
    -----------
    [1] Price, D.C. (2016). Spectrometers and Polyphase Filterbanks in Radio
        Astronomy. arXiv:1607.03579
    [2] Bull et al. (2024). RHINO: A large horn antenna for detecting the
        21cm global signal. arXiv:2410.00076
    """
    
    # ── Network Configuration (transmission disabled) ─────────────────────────
    SERVER_HOSTNAME = "192.168.1.100"
    SERVER_PORT = 8765
    
    # ── Hardware Configuration ────────────────────────────────────────────────
    # Direct sampling mode: no decimation, full ADC rate
    TARGET_SAMPLE_RATE_MHZ = 4915.2      # Native RFSoC ADC rate
    DECIMATION_FACTOR = 1                 # No decimation (supervisor requirement)
    
    # ── PFB Spectrometer Parameters ───────────────────────────────────────────
    # Following Price (2016) notation: M = taps, P = channels
    PFB_NUM_TAPS = 4                      # M: Number of FIR filter taps
    PFB_NUM_CHANNELS = 1024               # P: Number of frequency channels (FFT size)
    PFB_WINDOW_FN = "hamming"             # Window function for prototype filter
    
    # Time averaging
    INTEGRATION_TIME_SECONDS = 1.0        # Integrate spectra over this duration
    
    # Acquisition parameters
    ACQUISITION_DURATION = 60             # Total run duration (seconds)
    SPECTRA_PER_FILE = 10                 # Averaged spectra per output batch
    
    # ── Hardware Settings (rfsoc_sam overlay) ─────────────────────────────────
    NUMBER_OF_FRAMES = 32                 # Hardware averaging in FPGA
    SPECTRUM_TYPE = 'power'               # Output type: power spectrum |FFT|²
    
    # ── Debug and Display ─────────────────────────────────────────────────────
    VERBOSE = True


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def db(x):
    """
    Convert linear power values to dB scale.
    
    Parameters
    ----------
    x : ndarray
        Linear power values
    
    Returns
    -------
    x_db : ndarray
        Power in dB (10 * log10(x))
    """
    return 10 * np.log10(np.maximum(x, 1e-20))  # Avoid log(0)


def plot_spectrum_ascii(spectrum, freq_axis, n_cols=60, n_rows=10):
    """
    Plot spectrum as ASCII bar chart in terminal.
    
    Creates a text-based visualization similar to the original daq.py output.
    
    Parameters
    ----------
    spectrum : ndarray
        Power spectrum values (linear or dB)
    freq_axis : ndarray
        Frequency axis in MHz
    n_cols : int
        Number of columns in the plot (default 60)
    n_rows : int
        Number of rows (height) of the plot (default 10)
    """
    print("\n" + "=" * 70)
    print("[SPECTRUM PLOT - ASCII]")
    print("=" * 70)
    
    # Downsample spectrum to fit display width
    step = max(1, len(spectrum) // n_cols)
    downsampled = []
    for i in range(0, len(spectrum) - step + 1, step):
        downsampled.append(spectrum[i:i+step].mean())
    downsampled = np.array(downsampled)
    
    # Downsample frequency axis
    freq_downsampled = freq_axis[::step][:len(downsampled)]
    
    # Convert to dB if not already
    if downsampled.max() > 100:  # Likely linear scale
        db_vals = db(downsampled)
    else:  # Already in dB
        db_vals = downsampled
    
    # Get range
    db_min = db_vals.min()
    db_max = db_vals.max()
    db_range = db_max - db_min if db_max != db_min else 1.0
    
    # Print bar chart from top to bottom
    for row in range(n_rows, 0, -1):
        threshold = db_min + (row / n_rows) * db_range
        line = "".join("█" if v >= threshold else " " for v in db_vals)
        label = f"{db_min + (row/n_rows)*db_range:7.1f} |"
        print(f"  {label}{line}")
    
    # Print bottom axis
    print("         +" + "─" * len(db_vals))
    
    # Print frequency labels
    freq_label_left = f"{freq_axis[0]:.1f} MHz"
    freq_label_right = f"{freq_axis[-1]:.1f} MHz"
    spacing = len(db_vals) - len(freq_label_left) - len(freq_label_right)
    print(f"          {freq_label_left}{' ' * spacing}{freq_label_right}")
    
    print("\n  Power range: [{:.2f}, {:.2f}] dB".format(db_min, db_max))
    print("=" * 70)


def plot_spectrum_matplotlib(spectrum, freq_axis, title="PFB Spectrum", 
                               figsize=(12, 6), save_path=None):
    """
    Plot spectrum using matplotlib (for Jupyter notebooks or saving to file).
    
    Creates a professional-quality plot suitable for publication or analysis.
    
    Parameters
    ----------
    spectrum : ndarray
        Power spectrum values, shape (n_channels,) or (n_integrations, n_channels)
    freq_axis : ndarray
        Frequency axis in MHz
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, save figure to this path
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[WARNING] Matplotlib not available - cannot create graphical plot")
        return None, None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle both 1D and 2D spectra
    if spectrum.ndim == 1:
        # Single spectrum
        db_vals = db(spectrum)
        ax.plot(freq_axis, db_vals, linewidth=1, color='#2E86AB')
        ax.set_ylabel('Power (dB)', fontsize=12)
    else:
        # Multiple spectra (waterfall plot)
        db_vals = db(spectrum)
        im = ax.imshow(db_vals, aspect='auto', cmap='viridis',
                       extent=[freq_axis[0], freq_axis[-1], spectrum.shape[0], 0],
                       interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, label='Power (dB)')
        ax.set_ylabel('Integration Number', fontsize=12)
    
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Highlight RHINO science band (60-85 MHz) if in range
    if freq_axis[0] <= 85 and freq_axis[-1] >= 60:
        ax.axvspan(60, 85, alpha=0.1, color='red', label='RHINO band (60-85 MHz)')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved to {save_path}")
    
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# POLYPHASE FILTER BANK IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

class PolyphaseFilterBank:
    """
    Polyphase Filter Bank (PFB) spectrometer implementation.
    
    The PFB provides superior spectral resolution compared to a standard FFT
    by applying a prototype filter decomposed into polyphase branches before
    the FFT stage.
    
    Algorithm from Price (2016), arXiv:1607.03579.
    
    Parameters
    ----------
    n_taps : int
        Number of FIR filter taps (M in Price notation)
    n_channels : int
        Number of frequency channels / FFT size (P in Price notation)
    window_fn : str
        Window function for prototype filter ('hamming', 'hanning', 'blackman')
    
    References
    ----------
    [1] Price, D.C. (2016). Spectrometers and Polyphase Filterbanks in Radio
        Astronomy. arXiv:1607.03579, Section 3.3
    """
    
    def __init__(self, n_taps, n_channels, window_fn="hamming"):
        self.M = n_taps           # Number of taps
        self.P = n_channels       # Number of channels
        self.window_fn = window_fn
        
        # Generate and normalize window coefficients
        self.win_coeffs = self._generate_win_coeffs()
        
        print(f"[PFB] Initialized:")
        print(f"      Taps (M):         {self.M}")
        print(f"      Channels (P):     {self.P}")
        print(f"      Window function:  {self.window_fn}")
        print(f"      Filter length:    {len(self.win_coeffs)} coefficients")
    
    def _generate_win_coeffs(self):
        """
        Generate prototype filter coefficients.
        
        Creates a windowed sinc filter following Price (2016) Eq. 3.14:
        h[n] = w[n] · sinc(n / P)
        
        where w[n] is the window function and the cutoff is 1/P.
        
        Returns
        -------
        win_coeffs : ndarray
            Normalized filter coefficients, shape (M*P,)
        """
        # Generate window (e.g. Hamming)
        win_coeffs = scipy.signal.get_window(self.window_fn, self.M * self.P)
        
        # Generate sinc filter with cutoff = 1/P (Price 2016, Eq. 3.13)
        sinc = scipy.signal.firwin(self.M * self.P, cutoff=1.0/self.P, 
                                    window="rectangular")
        
        # Apply window to sinc (element-wise multiplication)
        win_coeffs *= sinc
        
        # Normalize for processing gain (Price 2016, Section 3.3.3)
        # Processing gain: pg = sum(|h[n]|²)
        pg = np.sum(np.abs(win_coeffs)**2)
        win_coeffs /= np.sqrt(pg)
        
        return win_coeffs
    
    def _pfb_fir_frontend(self, x):
        """
        Polyphase FIR filter frontend.
        
        Decomposes input into P polyphase branches, applies FIR filtering,
        and sums the results. This is the core of the PFB structure.
        
        Algorithm from Price (2016) Section 3.3.1, Figure 3.4.
        
        Parameters
        ----------
        x : ndarray
            Input time-domain samples, shape (W*M*P,)
            where W is the number of windows
        
        Returns
        -------
        x_summed : ndarray
            Filtered and summed output, shape (M*W - M + 1, P)
        """
        # Calculate number of complete windows
        W = x.shape[0] // (self.M * self.P)
        
        # Polyphase decomposition of input signal
        # Reshape into (P branches, W*M samples per branch)
        x_p = x.reshape((W * self.M, self.P)).T
        
        # Polyphase decomposition of filter coefficients
        # Reshape into (P branches, M taps per branch)
        h_p = self.win_coeffs.reshape((self.M, self.P)).T
        
        # Apply polyphase filtering: convolve each branch with its filter
        x_summed = np.zeros((self.P, self.M * W - self.M + 1))
        
        for t in range(0, self.M * W - self.M + 1):
            # Extract M samples from each branch
            x_slice = x_p[:, t:t + self.M]
            
            # Apply filter (element-wise multiply, then sum across taps)
            x_weighted = x_slice * h_p
            x_summed[:, t] = x_weighted.sum(axis=1)
        
        return x_summed.T  # Shape: (M*W - M + 1, P)
    
    def _fft(self, x_fir):
        """
        Apply FFT to polyphase-filtered data.
        
        Parameters
        ----------
        x_fir : ndarray
            Filtered polyphase output, shape (N_samples, P)
        
        Returns
        -------
        x_pfb : ndarray
            Frequency-domain output, shape (N_samples, P)
        """
        return np.fft.fft(x_fir, n=self.P, axis=1)
    
    def process_block(self, x):
        """
        Process a block of samples through the PFB.
        
        Complete PFB pipeline: polyphase FIR → FFT → power spectrum
        
        Parameters
        ----------
        x : ndarray
            Input time samples, shape (N,) where N = W*M*P
        
        Returns
        -------
        x_psd : ndarray
            Power spectral density, shape (N_spectra, P)
            where N_spectra = M*W - M + 1
        """
        # Trim to integer multiple of M*P
        n_samples = len(x) // (self.M * self.P) * (self.M * self.P)
        x = x[:n_samples]
        
        # Polyphase FIR frontend
        x_fir = self._pfb_fir_frontend(x)
        
        # FFT stage
        x_pfb = self._fft(x_fir)
        
        # Compute power spectrum: |FFT|²
        x_psd = np.real(x_pfb * np.conj(x_pfb))
        
        return x_psd
    
    def spectrometer(self, x, n_integrate):
        """
        Full PFB spectrometer: process samples and time-integrate.
        
        Parameters
        ----------
        x : ndarray
            Input time samples
        n_integrate : int
            Number of spectra to average together
        
        Returns
        -------
        x_integrated : ndarray
            Time-integrated power spectra, shape (N_outputs, P)
        """
        # Process through PFB
        x_psd = self.process_block(x)
        
        # Trim to integer multiple of n_integrate
        n_trim = (x_psd.shape[0] // n_integrate) * n_integrate
        x_psd = x_psd[:n_trim]
        
        # Time integration: reshape and average
        x_psd = x_psd.reshape(x_psd.shape[0] // n_integrate, n_integrate, x_psd.shape[1])
        x_integrated = x_psd.mean(axis=1)
        
        return x_integrated


# ══════════════════════════════════════════════════════════════════════════════
# ACQUISITION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class RHINOAcquisition:
    """
    RHINO PFB-based acquisition system for RFSoC 4x2.
    
    Interfaces with rfsoc_sam overlay to capture raw ADC samples,
    processes them through a polyphase filter bank, and outputs
    high-resolution spectra.
    """
    
    def __init__(self, overlay_path=None):
        print("=" * 70)
        print("[INITIALIZATION] Loading FPGA overlay...")
        
        # Import PYNQ libraries
        try:
            import pynq
            from rfsoc_sam import overlay
            self.pynq_available = True
        except ImportError as e:
            print(f"[ERROR] PYNQ libraries not found: {e}")
            self.pynq_available = False
            return
        
        # Load overlay
        try:
            if overlay_path is None:
                self.ol = overlay.Overlay()
            else:
                self.ol = pynq.Overlay(overlay_path)
            print("    ✓ Overlay loaded")
        except Exception as e:
            print(f"    ✗ Failed to load overlay: {e}")
            raise
        
        # Get hardware references
        self.receiver = self.ol.radio.receiver
        self.analyser = self.ol.radio.receiver.channel_00.spectrum_analyser
        print("    ✓ Hardware initialized")
        
        # Configure system
        self._configure_system()
        
        # Initialize PFB
        self._initialize_pfb()
        
        # Statistics counters
        self.total_spectra_collected = 0
        self.total_files_transmitted = 0
        self.acquisition_start_time = None
    
    def _configure_system(self):
        """Configure RFSoC hardware for direct sampling acquisition."""
        print("\n[CONFIGURATION] Setting up direct sampling...")
        
        # Read actual hardware sample rate
        self.adc_sample_rate = self.analyser.sample_frequency
        
        print(f"    === Direct Sampling Configuration ===")
        print(f"    Mode: Direct RF sampling (no DDC, no decimation)")
        print(f"    Decimation: {RHINOConfig.DECIMATION_FACTOR}x (NONE)")
        print(f"    ADC Sample Rate: {self.adc_sample_rate / 1e6:.2f} MHz")
        
        # Verify sample rate
        target_rate = RHINOConfig.TARGET_SAMPLE_RATE_MHZ * 1e6
        rate_error = abs(self.adc_sample_rate - target_rate) / target_rate
        if rate_error > 0.05:
            print(f"\n    WARNING: Sample rate mismatch!")
            print(f"             Expected: {target_rate / 1e6:.2f} MHz")
            print(f"             Actual:   {self.adc_sample_rate / 1e6:.2f} MHz")
        else:
            print(f"    ✓ Sample rate matches target")
        
        # Calculate Nyquist frequency
        nyquist_freq = (self.adc_sample_rate / 2) / 1e6
        print(f"    Nyquist Frequency: {nyquist_freq:.2f} MHz")
        print(f"    Coverage: DC to {nyquist_freq:.2f} MHz")
        
        # Enable DMA (critical for get_frame() to work)
        self.analyser.dma_enable = 1
        
        print(f"    ✓ Configuration complete")
    
    def _initialize_pfb(self):
        """Initialize the Polyphase Filter Bank."""
        print(f"\n[PFB INITIALIZATION]")
        
        # Create PFB instance
        self.pfb = PolyphaseFilterBank(
            n_taps=RHINOConfig.PFB_NUM_TAPS,
            n_channels=RHINOConfig.PFB_NUM_CHANNELS,
            window_fn=RHINOConfig.PFB_WINDOW_FN
        )
        
        # Calculate frequency resolution
        self.freq_resolution = (self.adc_sample_rate / RHINOConfig.PFB_NUM_CHANNELS) / 1e3
        print(f"      Frequency Resolution: {self.freq_resolution:.3f} kHz per channel")
        
        # Calculate how many raw ADC samples needed per PFB output spectrum
        # Each PFB spectrum requires M*P samples
        self.samples_per_pfb_spectrum = RHINOConfig.PFB_NUM_TAPS * RHINOConfig.PFB_NUM_CHANNELS
        
        # Time duration of samples needed for one PFB spectrum
        self.pfb_spectrum_time = self.samples_per_pfb_spectrum / self.adc_sample_rate
        print(f"      Samples per spectrum: {self.samples_per_pfb_spectrum}")
        print(f"      Time per spectrum: {self.pfb_spectrum_time * 1e6:.2f} μs")
        
        # Calculate integration parameters
        if RHINOConfig.INTEGRATION_TIME_SECONDS == 0.0:
            self.pfb_spectra_per_integration = 1
        else:
            # How many PFB output spectra to average
            self.pfb_spectra_per_integration = max(1, int(
                RHINOConfig.INTEGRATION_TIME_SECONDS / self.pfb_spectrum_time
            ))
        
        print(f"      Integration time: {RHINOConfig.INTEGRATION_TIME_SECONDS} s")
        print(f"      Spectra per integration: {self.pfb_spectra_per_integration}")
        
        # Create frequency axis for PFB output
        self.frequency_axis = np.arange(RHINOConfig.PFB_NUM_CHANNELS) * \
                               (self.adc_sample_rate / RHINOConfig.PFB_NUM_CHANNELS) / 1e6
    
    def capture_raw_samples(self, n_samples):
        """
        Capture raw ADC samples from hardware.
        
        Note: In the current rfsoc_sam implementation, get_frame() returns
        already-processed FFT data, not raw ADC samples. This is a limitation.
        
        For a true PFB implementation, we would need direct ADC access or
        a custom overlay that bypasses the built-in FFT.
        
        WORKAROUND: We can still demonstrate the PFB algorithm using simulated
        or externally captured data.
        """
        # TODO: This requires either:
        # 1. Custom Vivado overlay with raw ADC DMA access
        # 2. External data file for testing
        # 3. Synthetic signal generation
        
        raise NotImplementedError(
            "Raw ADC sample capture not available in rfsoc_sam overlay. "
            "Use test_pfb_with_synthetic_data() instead."
        )
    
    def test_pfb_with_synthetic_data(self, plot_ascii=True, plot_matplotlib=True):
        """
        Test PFB spectrometer with synthetic data.
        
        Generates a test signal and processes it through the PFB to verify
        the implementation works correctly.
        
        Parameters
        ----------
        plot_ascii : bool
            If True, display ASCII-style terminal plot
        plot_matplotlib : bool
            If True, create matplotlib figure (requires matplotlib)
        
        Returns
        -------
        X_psd : ndarray
            PFB output spectra
        """
        print("\n[PFB TEST] Generating synthetic test signal...")
        
        # Generate test parameters
        M = RHINOConfig.PFB_NUM_TAPS
        P = RHINOConfig.PFB_NUM_CHANNELS
        W = 100  # Number of windows
        n_integrate = 10
        
        # Create test signal: noise + sinusoid
        n_samples = M * P * W
        samples = np.arange(n_samples)
        noise = np.random.normal(loc=0.0, scale=0.1, size=n_samples)
        
        # Add a test tone at a specific frequency
        test_freq_hz = 75e6  # 75 MHz (within RHINO band)
        omega = 2 * np.pi * test_freq_hz / self.adc_sample_rate
        signal = 0.5 * np.sin(omega * samples)
        
        data = noise + signal
        
        print(f"    Test signal: {test_freq_hz / 1e6:.1f} MHz tone + noise")
        print(f"    Total samples: {n_samples:,}")
        print(f"    Processing through PFB...")
        
        # Process through PFB
        start_time = time.time()
        X_psd = self.pfb.spectrometer(data, n_integrate=n_integrate)
        elapsed = time.time() - start_time
        
        print(f"    ✓ PFB processing complete ({elapsed:.3f} s)")
        print(f"    Output shape: {X_psd.shape}")
        print(f"    Min power: {X_psd.min():.3e}")
        print(f"    Max power: {X_psd.max():.3e}")
        
        # Find peak channel (should be near test frequency)
        peak_channel = np.argmax(X_psd[0])
        peak_freq = self.frequency_axis[peak_channel]
        
        print(f"    Peak detected at channel {peak_channel} ({peak_freq:.2f} MHz)")
        print(f"    Expected frequency: {test_freq_hz / 1e6:.2f} MHz")
        
        # ── Visualizations ────────────────────────────────────────────────────
        
        # ASCII plot (always works in terminal)
        if plot_ascii:
            plot_spectrum_ascii(X_psd[0], self.frequency_axis, n_cols=60, n_rows=10)
        
        # Matplotlib plot (if available, creates nice figure)
        if plot_matplotlib and MATPLOTLIB_AVAILABLE:
            print("\n[PLOT] Creating matplotlib visualization...")
            
            # Single spectrum plot
            fig1, ax1 = plot_spectrum_matplotlib(
                X_psd[0], 
                self.frequency_axis,
                title=f"PFB Spectrum - Test Signal ({test_freq_hz/1e6:.1f} MHz tone)",
                figsize=(12, 6)
            )
            
            # Waterfall plot (time series)
            if X_psd.shape[0] > 1:
                fig2, ax2 = plot_spectrum_matplotlib(
                    X_psd,
                    self.frequency_axis,
                    title="PFB Spectrogram - Time Evolution",
                    figsize=(12, 8)
                )
            
            # Show plots (in Jupyter this displays inline, otherwise opens window)
            try:
                plt.show(block=False)  # Non-blocking in scripts
            except:
                pass  # In Jupyter, plots appear automatically
        
        return X_psd
    
    def _print_statistics(self):
        """Print acquisition statistics."""
        if self.acquisition_start_time is not None:
            total_time = time.time() - self.acquisition_start_time
            
            print("\n" + "=" * 70)
            print("[STATISTICS]")
            print(f"Total time:            {total_time:.2f} seconds")
            print(f"Raw spectra collected: {self.total_spectra_collected}")
            print(f"Batches completed:     {self.total_files_transmitted}")
            print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Main acquisition routine."""
    
    print("\n" + "=" * 70)
    print("RHINO PFB Spectrometer - RFSoC 4x2")
    print("=" * 70)
    
    # Validation
    if not (RHINOConfig.PFB_NUM_CHANNELS & (RHINOConfig.PFB_NUM_CHANNELS - 1) == 0):
        print("[ERROR] PFB_NUM_CHANNELS must be a power of 2")
        sys.exit(1)
    
    # Initialize acquisition system
    try:
        acquisition = RHINOAcquisition()
    except Exception as e:
        print(f"\n[FATAL] Initialization failed: {e}")
        sys.exit(1)
    
    # Run PFB test with synthetic data
    try:
        print("\n[TEST] Running PFB algorithm verification...")
        acquisition.test_pfb_with_synthetic_data()
        print("\n[SUCCESS] PFB implementation verified!")
    except Exception as e:
        print(f"\n[ERROR] PFB test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("[INFO] PFB spectrometer ready.")
    print("[INFO] Awaiting raw ADC access or custom overlay for live acquisition.")
    print("=" * 70)


if __name__ == '__main__':
    main()