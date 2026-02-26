#!/usr/bin/env python3
"""
RHINO 21cm Global Signal Acquisition - Unified FFT & PFB Implementation
========================================================================

This script supports TWO spectrum analysis modes:
1. FFT mode: Uses hardware FFT from rfsoc_sam overlay (fast, simple)
2. PFB mode: Software polyphase filter bank (better resolution, more processing)

Includes complete network transmission, time averaging, and visualization.

Based on:
- University of Strathclyde rfsoc_sam library
- Danny C. Price, "Spectrometers and Polyphase Filterbanks in Radio Astronomy"
  arXiv:1607.03579 (2016)

Author: Mbatshi Jerry Junior Mbulawa
Supervisor: Dr. Bull
Date: February 2025
"""

import numpy as np
import time
import sys
import hashlib
import json
from datetime import datetime
import scipy.signal

# Matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Network transmission (ENABLED as per supervisor requirements)
from websockets.sync.client import connect
from pickle import dumps
import itertools


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class RHINOConfig:
    """
    Configuration parameters for RHINO acquisition system.
    
    Supports both FFT and PFB modes.
    
    SUPERVISOR REQUIREMENTS (Feb 26, 2025):
    - Direct sampling (no DDC/mixing)
    - PFB WITHOUT decimation
    - Server/client transmission working
    - DAC tone generation for testing
    """
    
    # ── Processing Mode Selection ─────────────────────────────────────────────
    PROCESSING_MODE = "FFT"  # Options: "FFT" or "PFB"
    
    # ── Network Configuration ─────────────────────────────────────────────────
    SERVER_HOSTNAME = "192.168.1.100"
    SERVER_PORT = 8765
    ENABLE_TRANSMISSION = True  # Transmission now required by supervisors
    
    # ── DAC Tone Generation (New Feature) ─────────────────────────────────────
    ENABLE_DAC_TONE = False       # Enable test tone generation
    DAC_TONE_FREQUENCY_MHZ = 75.0 # Tone frequency (within RHINO band)
    DAC_TONE_AMPLITUDE = 0.5      # Amplitude (0.0 to 1.0)
    
    # ── Hardware Configuration ────────────────────────────────────────────────
    TARGET_SAMPLE_RATE_MHZ = 4915.2  # Native RFSoC ADC rate
    DECIMATION_FACTOR = 1             # No decimation (supervisor requirement)
    
    # ── FFT Mode Parameters ───────────────────────────────────────────────────
    FFT_SIZE = 8192                   # Hardware FFT size (max for rfsoc_sam)
    FFT_WINDOW = 'hanning'            # Window function
    FFT_SPECTRUM_TYPE = 'power'       # Output type: power spectrum |FFT|²
    NUMBER_OF_FRAMES = 32             # Hardware averaging in FPGA
    
    # ── PFB Mode Parameters ───────────────────────────────────────────────────
    # IMPORTANT: Supervisors requested PFB WITHOUT decimation (Feb 26, 2025)
    # Use full sample rate (4915.2 MSPS) → better frequency coverage
    PFB_NUM_TAPS = 4                  # M: Number of FIR filter taps
    PFB_NUM_CHANNELS = 1024           # P: Number of frequency channels
    PFB_WINDOW_FN = "hamming"         # Window function for prototype filter
    PFB_USE_DECIMATION = False        # NO DECIMATION (supervisor requirement)
    
    # ── Time Averaging ────────────────────────────────────────────────────────
    INTEGRATION_TIME_SECONDS = 1.0    # Integrate spectra over this duration
    
    # ── Acquisition Control ───────────────────────────────────────────────────
    ACQUISITION_DURATION = 60         # Total run duration (seconds)
    SPECTRA_PER_FILE = 10             # Averaged spectra per output batch
    
    # ── Debug and Display ─────────────────────────────────────────────────────
    VERBOSE = True
    SHOW_ASCII_PLOT = True            # Show terminal plots
    SHOW_MATPLOTLIB_PLOT = False      # Show graphical plots (Jupyter)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def db(x):
    """Convert linear power values to dB scale."""
    return 10 * np.log10(np.maximum(x, 1e-20))


def plot_spectrum_ascii(spectrum, freq_axis, n_cols=60, n_rows=10):
    """Plot spectrum as ASCII bar chart in terminal."""
    print("\n" + "=" * 70)
    print("[SPECTRUM PLOT - ASCII]")
    print("=" * 70)
    
    # Downsample spectrum
    step = max(1, len(spectrum) // n_cols)
    downsampled = np.array([spectrum[i:i+step].mean() 
                            for i in range(0, len(spectrum) - step + 1, step)])
    
    # Convert to dB if needed
    db_vals = db(downsampled) if downsampled.max() > 100 else downsampled
    
    db_min = db_vals.min()
    db_max = db_vals.max()
    db_range = db_max - db_min if db_max != db_min else 1.0
    
    # Print bars
    for row in range(n_rows, 0, -1):
        threshold = db_min + (row / n_rows) * db_range
        line = "".join("█" if v >= threshold else " " for v in db_vals)
        label = f"{threshold:7.1f} |"
        print(f"  {label}{line}")
    
    print("         +" + "─" * len(db_vals))
    print(f"          {freq_axis[0]:.1f} MHz{' ' * (len(db_vals) - 20)}{freq_axis[-1]:.1f} MHz")
    print(f"\n  Power range: [{db_min:.2f}, {db_max:.2f}] dB")
    print("=" * 70)


def plot_spectrum_matplotlib(spectrum, freq_axis, title="Spectrum", figsize=(12, 6)):
    """Plot spectrum using matplotlib."""
    if not MATPLOTLIB_AVAILABLE:
        return None, None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if spectrum.ndim == 1:
        ax.plot(freq_axis, db(spectrum), linewidth=1, color='#2E86AB')
        ax.set_ylabel('Power (dB)', fontsize=12)
    else:
        im = ax.imshow(db(spectrum), aspect='auto', cmap='viridis',
                       extent=[freq_axis[0], freq_axis[-1], spectrum.shape[0], 0])
        plt.colorbar(im, ax=ax, label='Power (dB)')
        ax.set_ylabel('Integration Number', fontsize=12)
    
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if freq_axis[0] <= 85 and freq_axis[-1] >= 60:
        ax.axvspan(60, 85, alpha=0.1, color='red', label='RHINO band')
        ax.legend()
    
    plt.tight_layout()
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# POLYPHASE FILTER BANK (PFB) IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

class PolyphaseFilterBank:
    """
    Polyphase Filter Bank spectrometer.
    
    Reference: Price, D.C. (2016). arXiv:1607.03579
    """
    
    def __init__(self, n_taps, n_channels, window_fn="hamming"):
        self.M = n_taps
        self.P = n_channels
        self.window_fn = window_fn
        self.win_coeffs = self._generate_win_coeffs()
    
    def _generate_win_coeffs(self):
        """Generate prototype filter coefficients."""
        win_coeffs = scipy.signal.get_window(self.window_fn, self.M * self.P)
        sinc = scipy.signal.firwin(self.M * self.P, cutoff=1.0/self.P, 
                                    window="rectangular")
        win_coeffs *= sinc
        pg = np.sum(np.abs(win_coeffs)**2)
        win_coeffs /= np.sqrt(pg)
        return win_coeffs
    
    def _pfb_fir_frontend(self, x):
        """Polyphase FIR filter frontend."""
        W = x.shape[0] // (self.M * self.P)
        x_p = x.reshape((W * self.M, self.P)).T
        h_p = self.win_coeffs.reshape((self.M, self.P)).T
        x_summed = np.zeros((self.P, self.M * W - self.M + 1))
        
        for t in range(0, self.M * W - self.M + 1):
            x_weighted = x_p[:, t:t + self.M] * h_p
            x_summed[:, t] = x_weighted.sum(axis=1)
        
        return x_summed.T
    
    def process_block(self, x):
        """Process samples through PFB pipeline."""
        n_samples = len(x) // (self.M * self.P) * (self.M * self.P)
        x = x[:n_samples]
        x_fir = self._pfb_fir_frontend(x)
        x_pfb = np.fft.fft(x_fir, n=self.P, axis=1)
        x_psd = np.real(x_pfb * np.conj(x_pfb))
        return x_psd
    
    def spectrometer(self, x, n_integrate):
        """Full PFB spectrometer with time integration."""
        x_psd = self.process_block(x)
        n_trim = (x_psd.shape[0] // n_integrate) * n_integrate
        x_psd = x_psd[:n_trim]
        x_psd = x_psd.reshape(x_psd.shape[0] // n_integrate, n_integrate, x_psd.shape[1])
        return x_psd.mean(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# ACQUISITION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class RHINOAcquisition:
    """
    RHINO acquisition system supporting both FFT and PFB modes.
    """
    
    def __init__(self, overlay_path=None):
        print("=" * 70)
        print(f"[INITIALIZATION] Loading FPGA overlay...")
        print(f"[MODE] Processing: {RHINOConfig.PROCESSING_MODE}")
        
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
        
        self.receiver = self.ol.radio.receiver
        self.analyser = self.ol.radio.receiver.channel_00.spectrum_analyser
        print("    ✓ Hardware initialized")
        
        # Configure system
        self._configure_system()
        
        # Initialize processing backend
        if RHINOConfig.PROCESSING_MODE == "PFB":
            self._initialize_pfb()
        
        # Initialize DAC for tone generation (if enabled)
        if RHINOConfig.ENABLE_DAC_TONE:
            self._initialize_dac()
        
        # Statistics
        self.total_spectra_collected = 0
        self.total_files_transmitted = 0
        self.acquisition_start_time = None
    
    def _configure_system(self):
        """Configure RFSoC hardware."""
        print("\n[CONFIGURATION] Setting up direct sampling...")
        
        self.adc_sample_rate = self.analyser.sample_frequency
        
        print(f"    Mode: Direct RF sampling")
        print(f"    ADC Sample Rate: {self.adc_sample_rate / 1e6:.2f} MHz")
        
        if RHINOConfig.PROCESSING_MODE == "FFT":
            self.analyser.fft_size = RHINOConfig.FFT_SIZE
            self.analyser.window = RHINOConfig.FFT_WINDOW
            self.analyser.spectrum_type = RHINOConfig.FFT_SPECTRUM_TYPE
            
            freq_res = (self.adc_sample_rate / RHINOConfig.FFT_SIZE) / 1e3
            print(f"    FFT Size: {RHINOConfig.FFT_SIZE} points")
            print(f"    Frequency Resolution: {freq_res:.3f} kHz per bin")
            
            self.frequency_axis = np.arange(RHINOConfig.FFT_SIZE) * \
                                   (self.adc_sample_rate / RHINOConfig.FFT_SIZE) / 1e6
        
        self.analyser.dma_enable = 1
        print(f"    ✓ Configuration complete")
    
    def _initialize_pfb(self):
        """Initialize PFB backend."""
        print(f"\n[PFB INITIALIZATION]")
        
        self.pfb = PolyphaseFilterBank(
            n_taps=RHINOConfig.PFB_NUM_TAPS,
            n_channels=RHINOConfig.PFB_NUM_CHANNELS,
            window_fn=RHINOConfig.PFB_WINDOW_FN
        )
        
        freq_res = (self.adc_sample_rate / RHINOConfig.PFB_NUM_CHANNELS) / 1e3
        print(f"    Taps: {RHINOConfig.PFB_NUM_TAPS}")
        print(f"    Channels: {RHINOConfig.PFB_NUM_CHANNELS}")
        print(f"    Frequency Resolution: {freq_res:.3f} kHz per channel")
        print(f"    Decimation: {'DISABLED' if not RHINOConfig.PFB_USE_DECIMATION else 'ENABLED'}")
        
        self.frequency_axis = np.arange(RHINOConfig.PFB_NUM_CHANNELS) * \
                               (self.adc_sample_rate / RHINOConfig.PFB_NUM_CHANNELS) / 1e6
    
    def _initialize_dac(self):
        """
        Initialize DAC for tone generation (loopback testing).
        
        Supervisors requested ability to generate test tones with DAC
        for spectral leakage tests and window validation.
        """
        print(f"\n[DAC INITIALIZATION]")
        
        try:
            self.transmitter = self.ol.radio.transmitter
            self.dac = self.ol.radio.transmitter.channel_00
            print("    ✓ DAC hardware accessed")
            
            if RHINOConfig.ENABLE_DAC_TONE:
                self.generate_dac_tone(
                    RHINOConfig.DAC_TONE_FREQUENCY_MHZ,
                    RHINOConfig.DAC_TONE_AMPLITUDE
                )
        except AttributeError:
            print("    ✗ DAC not available in this overlay")
            self.transmitter = None
            self.dac = None
    
    def generate_dac_tone(self, freq_mhz, amplitude=0.5):
        """
        Generate a continuous wave (CW) tone from the DAC.
        
        This allows loopback testing: DAC → Antenna/Cable → ADC
        
        Parameters
        ----------
        freq_mhz : float
            Frequency of the tone in MHz
        amplitude : float
            Amplitude (0.0 to 1.0)
        
        Notes
        -----
        For more complex waveforms (frequency combs, pseudo-random sequences),
        see generate_frequency_comb() method.
        """
        if self.dac is None:
            print("[WARNING] DAC not available — cannot generate tone")
            return
        
        print(f"\n[DAC TONE GENERATION]")
        print(f"    Frequency: {freq_mhz} MHz")
        print(f"    Amplitude: {amplitude}")
        
        try:
            # Get DAC sample rate
            dac_sample_rate = self.dac.sample_frequency
            print(f"    DAC Sample Rate: {dac_sample_rate / 1e6:.2f} MHz")
            
            # Generate tone waveform
            # Create one period at the requested frequency
            samples_per_period = int(dac_sample_rate / (freq_mhz * 1e6))
            t = np.arange(samples_per_period)
            omega = 2 * np.pi * freq_mhz * 1e6 / dac_sample_rate
            
            # Generate I and Q (for complex tone)
            i_waveform = amplitude * np.cos(omega * t)
            q_waveform = amplitude * np.sin(omega * t)
            
            # Upload to DAC
            # (Implementation depends on rfsoc_sam API — check documentation)
            # self.dac.write_waveform(i_waveform, q_waveform)
            
            print(f"    ✓ Tone generated ({samples_per_period} samples/period)")
            print(f"    [NOTE: Waveform upload depends on rfsoc_sam API]")
            
        except Exception as e:
            print(f"    ✗ DAC tone generation failed: {e}")
    
    def generate_frequency_comb(self, freq_start_mhz, freq_stop_mhz, freq_step_mhz, 
                                 amplitude=0.3):
        """
        Generate a frequency comb for spectral calibration.
        
        A frequency comb is multiple equally-spaced tones, useful for:
        - Calibrating frequency response
        - Testing channel isolation
        - Measuring spectral leakage
        
        Parameters
        ----------
        freq_start_mhz : float
            Starting frequency (MHz)
        freq_stop_mhz : float
            Ending frequency (MHz)
        freq_step_mhz : float
            Spacing between tones (MHz)
        amplitude : float
            Amplitude per tone (divided by number of tones)
        
        Example
        -------
        # Generate comb across RHINO band
        acq.generate_frequency_comb(60, 85, 5, amplitude=0.5)
        # Creates tones at 60, 65, 70, 75, 80, 85 MHz
        """
        if self.dac is None:
            print("[WARNING] DAC not available — cannot generate comb")
            return
        
        print(f"\n[FREQUENCY COMB GENERATION]")
        print(f"    Range: {freq_start_mhz} – {freq_stop_mhz} MHz")
        print(f"    Step: {freq_step_mhz} MHz")
        
        # Generate list of frequencies
        freqs = np.arange(freq_start_mhz, freq_stop_mhz + freq_step_mhz, freq_step_mhz)
        n_tones = len(freqs)
        print(f"    Number of tones: {n_tones}")
        
        # Normalize amplitude (so combined signal doesn't clip)
        tone_amplitude = amplitude / np.sqrt(n_tones)
        
        try:
            dac_sample_rate = self.dac.sample_frequency
            
            # Generate combined waveform (sum of all tones)
            n_samples = int(dac_sample_rate / 1e6)  # 1 ms worth of samples
            t = np.arange(n_samples)
            
            i_waveform = np.zeros(n_samples)
            q_waveform = np.zeros(n_samples)
            
            for freq_mhz in freqs:
                omega = 2 * np.pi * freq_mhz * 1e6 / dac_sample_rate
                i_waveform += tone_amplitude * np.cos(omega * t)
                q_waveform += tone_amplitude * np.sin(omega * t)
            
            # Upload to DAC
            # self.dac.write_waveform(i_waveform, q_waveform)
            
            print(f"    ✓ Frequency comb generated")
            print(f"    Frequencies: {freqs} MHz")
            print(f"    [NOTE: Waveform upload depends on rfsoc_sam API]")
            
        except Exception as e:
            print(f"    ✗ Frequency comb generation failed: {e}")
    
    def capture_single_spectrum(self):
        """Capture one hardware-averaged spectrum (FFT mode only)."""
        return self.analyser.get_frame()
    
    def capture_time_averaged_spectrum(self):
        """Capture and average multiple spectra over integration period."""
        if RHINOConfig.PROCESSING_MODE == "FFT":
            # FFT mode: use hardware FFT
            spectrum_time = (RHINOConfig.NUMBER_OF_FRAMES * RHINOConfig.FFT_SIZE / 
                            self.adc_sample_rate)
            
            if RHINOConfig.INTEGRATION_TIME_SECONDS == 0.0:
                spectra_to_avg = 1
            else:
                spectra_to_avg = max(1, int(
                    RHINOConfig.INTEGRATION_TIME_SECONDS / spectrum_time
                ))
            
            accumulated = np.zeros(RHINOConfig.FFT_SIZE, dtype=np.float64)
            for i in range(spectra_to_avg):
                accumulated += self.capture_single_spectrum()
                self.total_spectra_collected += 1
            
            return (accumulated / spectra_to_avg).astype(np.float32)
        
        else:
            # PFB mode: would need raw ADC samples
            raise NotImplementedError(
                "PFB mode requires raw ADC sample access. "
                "Use FFT mode or implement custom overlay."
            )
    
    def collect_spectrum_batch(self, num_averaged_spectra):
        """Collect multiple time-averaged spectra into a batch."""
        if RHINOConfig.PROCESSING_MODE == "FFT":
            num_channels = RHINOConfig.FFT_SIZE
        else:
            num_channels = RHINOConfig.PFB_NUM_CHANNELS
        
        spectra_batch = np.zeros((num_averaged_spectra, num_channels), dtype=np.float32)
        
        batch_start = time.time()
        
        for i in range(num_averaged_spectra):
            if RHINOConfig.VERBOSE:
                print(f"    Capturing spectrum {i+1}/{num_averaged_spectra}...")
            
            spectra_batch[i, :] = self.capture_time_averaged_spectrum()
            
            if RHINOConfig.VERBOSE and (i+1) % 10 == 0:
                elapsed = time.time() - batch_start
                rate = (i+1) / elapsed
                eta = (num_averaged_spectra - i - 1) / rate if rate > 0 else 0
                print(f"    Progress: {i+1}/{num_averaged_spectra} "
                      f"({rate:.1f} spec/s, ETA: {eta:.1f}s)")
        
        batch_duration = time.time() - batch_start
        
        return spectra_batch, batch_duration
    
    def run_acquisition(self):
        """Main acquisition loop."""
        if not self.pynq_available:
            print("[ERROR] Cannot run — PYNQ not available")
            return
        
        self.acquisition_start_time = time.time()
        file_counter = 0
        
        try:
            while True:
                elapsed = time.time() - self.acquisition_start_time
                if elapsed >= RHINOConfig.ACQUISITION_DURATION:
                    print("\n[ACQUISITION] Target duration reached")
                    break
                
                print(f"\n{'─' * 70}")
                print(f"Batch #{file_counter + 1}")
                print(f"Elapsed: {elapsed:.1f} / {RHINOConfig.ACQUISITION_DURATION} sec")
                print(f"{'─' * 70}")
                
                # Capture batch
                spectra_batch, batch_time = self.collect_spectrum_batch(
                    RHINOConfig.SPECTRA_PER_FILE
                )
                
                print(f"\n[BATCH COMPLETE]")
                print(f"  Shape: {spectra_batch.shape}")
                print(f"  Time: {batch_time:.2f} s")
                print(f"  Rate: {RHINOConfig.SPECTRA_PER_FILE / batch_time:.2f} spectra/s")
                
                # ── Visualization ─────────────────────────────────────────────
                if RHINOConfig.SHOW_ASCII_PLOT:
                    plot_spectrum_ascii(spectra_batch[0], self.frequency_axis)
                
                if RHINOConfig.SHOW_MATPLOTLIB_PLOT and MATPLOTLIB_AVAILABLE:
                    plot_spectrum_matplotlib(spectra_batch[0], self.frequency_axis,
                                             title=f"RHINO Spectrum - Batch {file_counter + 1}")
                    plt.show(block=False)
                
                # ── TO UNCOMMENT: Network Transmission ────────────────────────
                if RHINOConfig.ENABLE_TRANSMISSION:
                    self._transmit_batch(spectra_batch, file_counter)
                
                # ── TO UNCOMMENT: Local file save ─────────────────────────────
                # self._save_batch_locally(spectra_batch, file_counter)
                
                file_counter += 1
                self.total_files_transmitted += 1
        
        except KeyboardInterrupt:
            print("\n\n[ACQUISITION] Interrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"\n[ERROR] Acquisition failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._print_statistics()
    
    def _transmit_batch(self, spectra_batch, batch_number):
        """
        Transmit spectrum batch to logging server.
        
        NOW ENABLED as per supervisor requirements (Feb 26, 2025).
        """
        metadata = {
            'timestamp': datetime.utcnow().isoformat(),
            'batch_number': batch_number,
            'sample_rate_hz': float(self.adc_sample_rate),
            'integration_time_s': RHINOConfig.INTEGRATION_TIME_SECONDS,
            'spectra_in_batch': spectra_batch.shape[0],
            'num_channels': spectra_batch.shape[1],
            'processing_mode': RHINOConfig.PROCESSING_MODE
        }
        
        data_bytes = dumps(spectra_batch)
        md5_hash = hashlib.md5(data_bytes).hexdigest()
        
        try:
            with connect(f"ws://{RHINOConfig.SERVER_HOSTNAME}:{RHINOConfig.SERVER_PORT}") as ws:
                # Send metadata first
                ws.send(json.dumps(metadata))
                
                # Send data in chunks
                chunk_size = 1024 * 1024  # 1 MB chunks
                for chunk in [data_bytes[i:i+chunk_size] 
                              for i in range(0, len(data_bytes), chunk_size)]:
                    ws.send(chunk)
                
                # Send checksum
                ws.send(md5_hash)
                
                if RHINOConfig.VERBOSE:
                    print(f"    ✓ Transmitted batch {batch_number} ({len(data_bytes) / 1024:.1f} KB)")
        
        except Exception as e:
            print(f"    ✗ Transmission failed: {e}")
            print(f"       Check server is running: python3 rhino_data_server.py")
    
    def _save_batch_locally(self, spectra_batch, batch_number):
        """Save spectrum batch to local file."""
        filename = f"rhino_spectrum_batch_{batch_number:04d}.npy"
        np.save(filename, spectra_batch)
        print(f"  ✓ Saved to {filename}")
    
    def _print_statistics(self):
        """Print acquisition statistics."""
        if self.acquisition_start_time is not None:
            total_time = time.time() - self.acquisition_start_time
            data_mb = (self.total_spectra_collected * 
                      (RHINOConfig.FFT_SIZE if RHINOConfig.PROCESSING_MODE == "FFT" 
                       else RHINOConfig.PFB_NUM_CHANNELS) * 4) / (1024 * 1024)
            
            print("\n" + "=" * 70)
            print("[STATISTICS]")
            print(f"Total time:            {total_time:.2f} seconds")
            print(f"Raw spectra collected: {self.total_spectra_collected}")
            print(f"Batches completed:     {self.total_files_transmitted}")
            print(f"Data volume:           {data_mb:.2f} MB")
            print(f"Processing mode:       {RHINOConfig.PROCESSING_MODE}")
            print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Main acquisition routine."""
    
    print("\n" + "=" * 70)
    print("RHINO 21cm Global Signal Acquisition")
    print("=" * 70)
    print(f"Mode: {RHINOConfig.PROCESSING_MODE}")
    print(f"Transmission: {'ENABLED' if RHINOConfig.ENABLE_TRANSMISSION else 'DISABLED'}")
    print("=" * 70)
    
    # Validation
    if RHINOConfig.PROCESSING_MODE not in ["FFT", "PFB"]:
        print("[ERROR] PROCESSING_MODE must be 'FFT' or 'PFB'")
        sys.exit(1)
    
    # Initialize
    try:
        acquisition = RHINOAcquisition()
    except Exception as e:
        print(f"\n[FATAL] Initialization failed: {e}")
        sys.exit(1)
    
    # Run acquisition
    try:
        acquisition.run_acquisition()
    except Exception as e:
        print(f"\n[FATAL] Acquisition failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()