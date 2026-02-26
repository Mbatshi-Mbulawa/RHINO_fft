#!/usr/bin/env python3
"""
RHINO 21cm Global Signal Acquisition - Direct Sampling Mode
==========================================================

Author: Mbatshi Jerry Junior Mbulawa
Project: RFSoC DSP and PYNQ - RHINO Receiver Platform
Supervisor: Dr. Bull
Institution: University of Manchester (Jodrell Bank Centre for Astrophysics)
Date: February 2026

Description:
-----------
This script implements spectrum acquisition for the RHINO (Remote HI eNvironment 
Observer) experiment, which aims to detect the 21cm global signal from neutral 
hydrogen during the Cosmic Dawn and Epoch of Reionisation (z ~ 15-23).

Key difference from standard RFSoC applications:
- **DIRECT SAMPLING**: No digital downconversion (DDC)
- **NO DECIMATION**: Full sample rate maintained (decimation = 1)
- **NO CENTRE FREQUENCY**: Observing 60-85 MHz band directly

This is different from typical radio astronomy, where you tune to a specific 
frequency and mix it down to baseband. Instead, we're sampling the RF directly.

Scientific Context (from Bull et al. 2024, arXiv:2410.00076):
------------------------------------------------------------
The 21cm global signal is the sky-averaged brightness temperature of neutral 
hydrogen across cosmic time. RHINO targets 60-85 MHz (z ~ 15.7-22.7), 
corresponding to the epoch when the first stars formed and began heating/ionising 
the intergalactic medium.

Expected signal characteristics:
- Amplitude: ~50-200 mK (millikelvin) absorption features
- Spectral structure: Smooth, ~10+ MHz wide features
- Foreground contamination: ~10^4-10^5 K (100-1000× brighter!)
- Key challenge: Separating faint signal from bright, smooth foregrounds

System Architecture:
-------------------
                                                     NETWORK
    ┌─────────────────────────────────────┐       (Ethernet)      ┌──────────────────────┐
    │       RFSoC 4x2 Board               │                        │  Logging Computer    │
    │                                     │                        │                      │
    │  ┌──────────┐                       │                        │  ┌────────────────┐  │
    │  │   ADC    │ NO DDC/MIXER!         │                        │  │  WebSocket     │  │
    │  │  Tile    │ Direct sampling       │                        │  │  Server        │  │
    │  └──────────┘ 60-85 MHz            │                        │  │  (Receiver)    │  │
    │        │                             │                        │  └────────────────┘  │
    │        v                             │                        │          │           │
    │  ┌──────────────────────────┐       │      Spectrum Data     │          v           │
    │  │   FFT Processing         │       │   ══════════════════>  │  ┌────────────────┐  │
    │  │   16384 points           │       │    (NumPy Arrays)      │  │  File Storage  │  │
    │  │   NO DECIMATION (1x)     │       │                        │  │  (.npy files)  │  │
    │  └──────────────────────────┘       │                        │  └────────────────┘  │
    │        │                             │                        │                      │
    │        v                             │                        └──────────────────────┘
    │  ┌──────────────────────────┐       │
    │  │   DMA → DDR Memory       │       │
    │  └──────────────────────────┘       │
    │        │                             │
    │        v                             │
    │  ┌──────────────────────────┐       │
    │  │ Python Control (PYNQ)    │       │
    │  │ - Configuration          │       │
    │  │ - Data Retrieval         │       │
    │  │ - Network Transfer       │       │
    │  └──────────────────────────┘       │
    └─────────────────────────────────────┘

References:
----------
[1] Bull et al. (2024), "RHINO: A large horn antenna for detecting the 21cm 
    global signal", arXiv:2410.00076
[2] Bowman et al. (2018), "An absorption profile centred at 78 megahertz in 
    the sky-averaged spectrum", Nature, 555, 67-70
[3] Pritchard & Loeb (2012), "21 cm cosmology in the 21st century", 
    Rep. Prog. Phys., 75, 086901

"""

import numpy as np
import time
import sys
import hashlib
import json
from datetime import datetime
from websockets.sync.client import connect
from pickle import dumps
import itertools

# =============================================================================
# CONFIGURATION PARAMETERS FOR RHINO DIRECT SAMPLING
# =============================================================================

class RHINOConfig:
    """
    Configuration parameters for RHINO 21cm global signal acquisition.
    
    CRITICAL: This configuration is for DIRECT SAMPLING mode.
    - No frequency mixing/downconversion
    - No decimation (sample at full ADC rate)
    - Observing 60-85 MHz band directly
    """
    
    # Server Configuration
    # -------------------
    SERVER_HOSTNAME = "192.168.1.100"  # IP address of logging computer
    SERVER_PORT = 8765                  # WebSocket server port
    WEBSOCKETS_MAX_SIZE = int(1e6)      # Maximum WebSocket message size (bytes)
    WEBSOCKETS_CHUNK_SIZE = int(1e6)    # Data chunk size for transmission (bytes)
    
    # RF Configuration - DIRECT SAMPLING MODE
    # ----------------------------------------
    # For RHINO, we are NOT using the digital downconverter (DDC).
    # We sample the RF band directly from DC to Nyquist frequency.
    
    # NO CENTRE FREQUENCY - we're not mixing/shifting frequency
    # The entire DC-150 MHz band is captured directly by the ADC
    
    # Target Sample Rate Configuration
    # --------------------------------
    # PhD student specification: 300 MHz sampling rate
    # This gives:
    #   - Nyquist frequency: 150 MHz
    #   - Coverage: DC to 150 MHz
    #   - RHINO band (60-85 MHz): Well within Nyquist
    #   - FM band (88-108 MHz): Properly sampled (no aliasing)
    #
    # Why 300 MHz and not higher?
    #   1. FM Contamination: FM band (88-108 MHz) is ~10^9× brighter than signal
    #   2. Aliasing Prevention: With Fs=170 MHz, FM would alias into 60-85 MHz band
    #   3. With Fs=300 MHz, FM is properly sampled and can be digitally filtered
    #   4. No need for higher Fs - nothing of interest above 150 MHz
    
    TARGET_SAMPLE_RATE_MHZ = 300.0  # Target ADC sample rate in MHz
    
    DECIMATION_FACTOR = 1  # NO DECIMATION - keep full sample rate
                           # Critical for maintaining bandwidth to 150 MHz
    
    # FFT Configuration
    # ----------------
    # PhD student specification: 16384 points (2^14) as starting point
    # Could increase for better channelization and FM isolation
    #
    # At 300 MHz sample rate:
    #   FFT_SIZE = 16384 → Frequency resolution = 300/16384 = 18.3 kHz per bin
    #   FFT_SIZE = 32768 → Frequency resolution = 300/32768 = 9.2 kHz per bin
    #   FFT_SIZE = 65536 → Frequency resolution = 300/65536 = 4.6 kHz per bin
    #
    # Larger FFT = Better frequency resolution = Better FM isolation
    #            = More processing time and memory
    
    FFT_SIZE = 16384       # Starting point (can increase to 32768 or 65536)
                           # Frequency resolution @ 300 MHz = 18.3 kHz per bin
    
    NUMBER_OF_FRAMES = 32  # Number of FFT frames to average (1-64)
                           # More frames = lower noise, but slower cadence
    
    WINDOW_TYPE = 'hanning'  # Window function to reduce spectral leakage
                             # Options: 'rectangular', 'hanning', 'hamming', 'blackman'
    
    SPECTRUM_TYPE = 'power'  # Output type: 'magnitude', 'power', 'log'
                            # 'power' is standard for radio astronomy (units: K^2)
    
    # Band Selection After FFT
    # ------------------------
    # After FFT, we'll extract two regions:
    # 1. RHINO science band: 60-85 MHz (primary target)
    # 2. FM contamination band: 88-108 MHz (for monitoring/flagging)
    
    # RHINO Science Band
    BAND_START_MHZ = 60.0    # Start of RHINO observing band
    BAND_END_MHZ = 85.0      # End of RHINO observing band
    
    # FM Contamination Band (for monitoring)
    FM_START_MHZ = 88.0      # Start of FM broadcast band
    FM_END_MHZ = 108.0       # End of FM broadcast band
    
    # Why track FM band?
    # - FM is 10^9× brighter than 21cm signal
    # - Sidelobes/filter rolloff can contaminate RHINO band
    # - Need to monitor FM power to flag contaminated spectra
    # - Can use FM variability to identify/remove systematic effects
    
    # Data Collection
    # --------------
    ACQUISITION_DURATION = 3600    # Total acquisition time in seconds (1 hour)
    SPECTRA_PER_FILE = 60          # Number of spectra per file (e.g., 1 per minute)
    
    # Calibration
    # ----------
    CALIBRATION_MODE = 1     # ADC calibration mode (1 or 2)
    
    # Data Quality
    # -----------
    ENABLE_MD5_CHECKSUM = True   # Compute MD5 checksums for data integrity
    
    # Logging
    # ------
    VERBOSE = True               # Enable detailed console output
    SAVE_LOCAL_COPY = False      # Save data locally before transmission
    LOCAL_DATA_DIR = "./data"    # Directory for local data storage


# =============================================================================
# DATA TRANSMISSION UTILITIES
# =============================================================================

def send_array(websocket, arr, chunk_size=RHINOConfig.WEBSOCKETS_CHUNK_SIZE):
    """
    Transmit a numpy array over WebSocket connection in chunks.
    
    [Same implementation as before - no changes needed]
    """
    bytes_arr = dumps(arr)
    
    if RHINOConfig.VERBOSE:
        print(f"    Serialized array size: {len(bytes_arr) / (1024*1024):.2f} MB")
    
    t_start = time.time()
    chunk_count = 0
    
    for chunk in itertools.batched(bytes_arr, chunk_size):
        websocket.send(bytes(chunk), text=False)
        chunk_count += 1
    
    transmission_time = time.time() - t_start
    
    if RHINOConfig.VERBOSE:
        print(f"    Transmitted {chunk_count} chunks in {transmission_time:.3f} seconds")
        print(f"    Effective throughput: {(len(bytes_arr)/(1024*1024))/transmission_time:.2f} MB/s")


def transmit_spectrum_data(spectrum_data, metadata):
    """
    Establish WebSocket connection and transmit spectrum data with metadata.
    
    [Same implementation as before - no changes needed]
    """
    server_url = f"ws://{RHINOConfig.SERVER_HOSTNAME}:{RHINOConfig.SERVER_PORT}"
    
    try:
        if RHINOConfig.VERBOSE:
            print(f"\n[TRANSMISSION] Connecting to server at {server_url}")
        
        with connect(server_url, max_size=RHINOConfig.WEBSOCKETS_MAX_SIZE) as websocket:
            
            if RHINOConfig.VERBOSE:
                print("    Sending metadata...")
            websocket.send(json.dumps(metadata), text=True)
            
            if RHINOConfig.VERBOSE:
                print("    Sending spectrum data...")
            send_array(websocket, spectrum_data)
            
            print(f"[TRANSMISSION] Successfully transmitted file: {metadata['filename']}")
            
    except ConnectionRefusedError:
        print(f"[ERROR] Cannot connect to server at {server_url}")
        print("        Ensure rhino_data_server.py is running on the logging computer")
        raise
    except Exception as e:
        print(f"[ERROR] Transmission failed: {e}")
        raise


# =============================================================================
# RHINO SPECTRUM ACQUISITION SYSTEM - DIRECT SAMPLING MODE
# =============================================================================

class RHINODirectSampling:
    """
    RHINO spectrum acquisition controller for direct sampling mode.
    
    Key differences from standard RFSoC operation:
    - No DDC (digital downconverter) - no frequency mixing
    - No decimation - full sample rate maintained
    - Direct FFT on RF signal
    - Band extraction after FFT to isolate 60-85 MHz
    """
    
    def __init__(self, overlay_path=None):
        """
        Initialize the RHINO direct sampling acquisition system.
        """
        print("="*70)
        print("RHINO 21cm Global Signal Acquisition System")
        print("Direct Sampling Mode (No DDC, No Decimation)")
        print("Jodrell Bank Centre for Astrophysics - University of Manchester")
        print("="*70)
        
        # Import PYNQ and rfsoc_sam modules
        try:
            import pynq
            from rfsoc_sam import overlay
            self.pynq_available = True
        except ImportError as e:
            print(f"[ERROR] Required PYNQ libraries not found: {e}")
            print("        This script must run on the RFSoC board with PYNQ installed")
            self.pynq_available = False
            return
        
        # Load the spectrum analyzer overlay
        print("\n[INITIALIZATION] Loading FPGA overlay...")
        try:
            if overlay_path is None:
                self.ol = overlay.SpectrumAnalyserOverlay()
            else:
                self.ol = pynq.Overlay(overlay_path)
            print("    ✓ Overlay loaded successfully")
        except Exception as e:
            print(f"    ✗ Failed to load overlay: {e}")
            raise
        
        # Get references to hardware components
        self.receiver = self.ol.radio.receiver
        
        print("    ✓ Hardware components initialized")
        
        # Configure for direct sampling
        self._configure_direct_sampling()
        
        # Initialize statistics
        self.total_spectra_collected = 0
        self.total_files_transmitted = 0
        self.acquisition_start_time = None
        
    
    def _configure_direct_sampling(self):
        """
        Configure RFSoC for direct sampling mode (no DDC, no decimation).
        
        Technical Details:
        -----------------
        In direct sampling mode:
        1. ADC samples RF directly at multi-GSPS rate
        2. No frequency mixing/translation occurs
        3. FFT operates on raw RF samples
        4. We extract the 60-85 MHz portion after FFT
        
        This is fundamentally different from channelized receivers that 
        use DDC to translate and decimate before processing.
        """
        print("\n[CONFIGURATION] Setting up direct sampling mode...")
        
        # CRITICAL: Set decimation to 1 (no decimation)
        print("\n    === DIRECT SAMPLING CONFIGURATION ===")
        print("    Mode: Direct RF sampling (no DDC)")
        self.receiver.analyser.decimation_factor = RHINOConfig.DECIMATION_FACTOR
        print(f"    Decimation Factor: {RHINOConfig.DECIMATION_FACTOR}x (NO DECIMATION)")
        
        # Read actual ADC sample rate
        self.adc_sample_rate = self.receiver.analyser.sample_frequency
        print(f"    ADC Sample Rate: {self.adc_sample_rate / 1e6:.2f} MHz")
        
        # Verify sample rate matches target (within tolerance)
        target_rate = RHINOConfig.TARGET_SAMPLE_RATE_MHZ * 1e6  # Hz
        rate_tolerance = 0.05  # 5% tolerance
        rate_error = abs(self.adc_sample_rate - target_rate) / target_rate
        
        if rate_error > rate_tolerance:
            print(f"\n    ⚠ WARNING: Sample rate mismatch!")
            print(f"      Target: {RHINOConfig.TARGET_SAMPLE_RATE_MHZ} MHz")
            print(f"      Actual: {self.adc_sample_rate/1e6:.2f} MHz")
            print(f"      Error: {rate_error*100:.1f}%")
            print(f"    Continuing anyway, but verify RFSoC configuration...")
        else:
            print(f"    ✓ Sample rate matches target ({RHINOConfig.TARGET_SAMPLE_RATE_MHZ} MHz)")
        
        # Calculate Nyquist frequency
        nyquist_freq = (self.adc_sample_rate / 2) / 1e6  # MHz
        print(f"    Nyquist Frequency: {nyquist_freq:.2f} MHz")
        
        # Verify we can capture our target band
        if nyquist_freq < RHINOConfig.BAND_END_MHZ:
            raise ValueError(
                f"ADC sample rate too low!\n"
                f"  Nyquist frequency: {nyquist_freq:.2f} MHz\n"
                f"  Target band: {RHINOConfig.BAND_START_MHZ}-{RHINOConfig.BAND_END_MHZ} MHz\n"
                f"  Need sample rate ≥ {2 * RHINOConfig.BAND_END_MHZ} MHz"
            )
        
        print(f"    ✓ Sample rate sufficient for {RHINOConfig.BAND_START_MHZ}-{RHINOConfig.BAND_END_MHZ} MHz band")
        
        # Set FFT size
        self.receiver.analyser.fftsize = RHINOConfig.FFT_SIZE
        freq_resolution = (self.adc_sample_rate / RHINOConfig.FFT_SIZE) / 1e3  # kHz
        print(f"\n    FFT Size: {RHINOConfig.FFT_SIZE} points")
        print(f"    Frequency Resolution: {freq_resolution:.3f} kHz/bin")
        
        # Calculate which FFT bins correspond to our 60-85 MHz band
        self._calculate_band_indices()
        
        # Set frame averaging
        self.receiver.analyser.number_frames = RHINOConfig.NUMBER_OF_FRAMES
        integration_time = (RHINOConfig.NUMBER_OF_FRAMES * 
                           RHINOConfig.FFT_SIZE / self.adc_sample_rate)
        print(f"\n    Frame Averaging: {RHINOConfig.NUMBER_OF_FRAMES} frames")
        print(f"    Integration Time: {integration_time*1e3:.2f} ms per spectrum")
        
        # Set window function
        self.receiver.analyser.window = RHINOConfig.WINDOW_TYPE
        print(f"    Window Function: {RHINOConfig.WINDOW_TYPE}")
        
        # Set spectrum type
        self.receiver.analyser.spectrum_type = RHINOConfig.SPECTRUM_TYPE
        print(f"    Spectrum Type: {RHINOConfig.SPECTRUM_TYPE}")
        
        # Set calibration mode
        self.receiver.analyser.calibration_mode = RHINOConfig.CALIBRATION_MODE
        print(f"    Calibration Mode: {RHINOConfig.CALIBRATION_MODE}")
        
        print("\n    ✓ Direct sampling configured successfully\n")
    
    
    def _calculate_band_indices(self):
        """
        Calculate which FFT bins correspond to:
        1. RHINO science band (60-85 MHz)
        2. FM contamination band (88-108 MHz)
        
        Technical Note:
        --------------
        FFT output is organized as:
        - Bins 0 to N/2: Frequencies 0 to Nyquist (positive frequencies)
        - Bins N/2+1 to N-1: Negative frequencies (usually ignored in RF)
        
        Example calculation for 300 MHz sampling:
        - Freq per bin: 300 MHz / 16384 = 18.3 kHz
        - 60 MHz → bin 3276
        - 85 MHz → bin 4642
        - 88 MHz → bin 4806
        - 108 MHz → bin 5898
        """
        # Frequency of each FFT bin
        freq_per_bin = self.adc_sample_rate / RHINOConfig.FFT_SIZE  # Hz
        
        # RHINO science band
        self.rhino_start_bin = int(RHINOConfig.BAND_START_MHZ * 1e6 / freq_per_bin)
        self.rhino_end_bin = int(RHINOConfig.BAND_END_MHZ * 1e6 / freq_per_bin)
        self.num_rhino_bins = self.rhino_end_bin - self.rhino_start_bin
        
        # FM contamination band
        self.fm_start_bin = int(RHINOConfig.FM_START_MHZ * 1e6 / freq_per_bin)
        self.fm_end_bin = int(RHINOConfig.FM_END_MHZ * 1e6 / freq_per_bin)
        self.num_fm_bins = self.fm_end_bin - self.fm_start_bin
        
        # Verify bins are within valid range
        nyquist_bin = RHINOConfig.FFT_SIZE // 2
        if self.fm_end_bin > nyquist_bin:
            raise ValueError(
                f"FM band extends beyond Nyquist frequency!\n"
                f"  FM band: {RHINOConfig.FM_START_MHZ}-{RHINOConfig.FM_END_MHZ} MHz\n"
                f"  Nyquist bin: {nyquist_bin} (freq: {(nyquist_bin * freq_per_bin)/1e6:.2f} MHz)\n"
                f"  Need sample rate ≥ {2 * RHINOConfig.FM_END_MHZ} MHz"
            )
        
        print(f"\n    === Band Configuration ===")
        print(f"    RHINO Science Band: {RHINOConfig.BAND_START_MHZ}-{RHINOConfig.BAND_END_MHZ} MHz")
        print(f"      FFT Bin Range: {self.rhino_start_bin} to {self.rhino_end_bin}")
        print(f"      Number of Bins: {self.num_rhino_bins}")
        print(f"      Actual Coverage: {(self.rhino_start_bin * freq_per_bin)/1e6:.3f} - {(self.rhino_end_bin * freq_per_bin)/1e6:.3f} MHz")
        
        print(f"\n    FM Contamination Band: {RHINOConfig.FM_START_MHZ}-{RHINOConfig.FM_END_MHZ} MHz")
        print(f"      FFT Bin Range: {self.fm_start_bin} to {self.fm_end_bin}")
        print(f"      Number of Bins: {self.num_fm_bins}")
        print(f"      Actual Coverage: {(self.fm_start_bin * freq_per_bin)/1e6:.3f} - {(self.fm_end_bin * freq_per_bin)/1e6:.3f} MHz")
        print(f"      ⚠ FM is ~10^9× brighter than 21cm signal!")
    
    
    def capture_single_spectrum(self):
        """
        Capture a single averaged spectrum from the RFSoC.
        
        Returns:
        -------
        spectrum_rhino : np.ndarray
            Extracted 60-85 MHz band (science target)
        spectrum_fm : np.ndarray
            Extracted 88-108 MHz band (contamination monitoring)
        freq_rhino : np.ndarray
            Frequency axis for RHINO band (in MHz)
        freq_fm : np.ndarray
            Frequency axis for FM band (in MHz)
        
        Technical Notes:
        ---------------
        We extract BOTH bands to:
        1. RHINO band: Primary science target
        2. FM band: Monitor contamination levels for data quality flagging
        
        The FM band is 10^9× brighter, so we can use its variability to:
        - Flag times when FM sidelobes contaminate RHINO band
        - Identify systematic effects correlated with FM power
        - Verify filter performance
        """
        # Capture full FFT spectrum
        spectrum_full = self.receiver.analyser.spectrum_data()
        
        # Extract RHINO science band (60-85 MHz)
        spectrum_rhino = spectrum_full[self.rhino_start_bin:self.rhino_end_bin]
        
        # Extract FM contamination band (88-108 MHz)
        spectrum_fm = spectrum_full[self.fm_start_bin:self.fm_end_bin]
        
        # Create frequency axes
        freq_per_bin = self.adc_sample_rate / RHINOConfig.FFT_SIZE  # Hz
        freq_rhino = np.arange(self.rhino_start_bin, self.rhino_end_bin) * freq_per_bin / 1e6  # MHz
        freq_fm = np.arange(self.fm_start_bin, self.fm_end_bin) * freq_per_bin / 1e6  # MHz
        
        return spectrum_rhino, spectrum_fm, freq_rhino, freq_fm
    
    
    def collect_spectra_batch(self, num_spectra):
        """
        Collect multiple consecutive spectra and return as 2D arrays.
        
        For RHINO, we store both:
        1. Science band (60-85 MHz) - primary target
        2. FM band (88-108 MHz) - contamination monitoring
        
        Returns:
        -------
        spectra_rhino : np.ndarray
            2D array of shape (num_spectra, num_rhino_bins)
        spectra_fm : np.ndarray
            2D array of shape (num_spectra, num_fm_bins)
        freq_rhino : np.ndarray
            Frequency axis for RHINO band (MHz)
        freq_fm : np.ndarray
            Frequency axis for FM band (MHz)
        """
        print(f"[ACQUISITION] Collecting batch of {num_spectra} spectra...")
        
        # Pre-allocate arrays
        spectra_rhino = np.zeros((num_spectra, self.num_rhino_bins), dtype=np.float32)
        spectra_fm = np.zeros((num_spectra, self.num_fm_bins), dtype=np.float32)
        
        batch_start_time = time.time()
        
        # Capture first spectrum to get frequency axes
        spectrum_rhino, spectrum_fm, freq_rhino, freq_fm = self.capture_single_spectrum()
        spectra_rhino[0, :] = spectrum_rhino
        spectra_fm[0, :] = spectrum_fm
        self.total_spectra_collected += 1
        
        # Capture remaining spectra
        for i in range(1, num_spectra):
            spectrum_rhino, spectrum_fm, _, _ = self.capture_single_spectrum()
            spectra_rhino[i, :] = spectrum_rhino
            spectra_fm[i, :] = spectrum_fm
            
            self.total_spectra_collected += 1
            
            if RHINOConfig.VERBOSE and (i+1) % 10 == 0:
                elapsed = time.time() - batch_start_time
                rate = (i+1) / elapsed
                
                # Calculate FM contamination level
                fm_mean_power = np.mean(spectra_fm[:i+1, :])
                rhino_mean_power = np.mean(spectra_rhino[:i+1, :])
                contamination_ratio = fm_mean_power / rhino_mean_power if rhino_mean_power > 0 else 0
                
                print(f"    Progress: {i+1}/{num_spectra} spectra ({rate:.1f} spectra/sec)")
                print(f"      FM/RHINO power ratio: {contamination_ratio:.2e}")
        
        batch_duration = time.time() - batch_start_time
        
        print(f"    ✓ Batch complete: {batch_duration:.2f} seconds")
        print(f"    Average capture rate: {num_spectra/batch_duration:.2f} spectra/sec")
        
        return spectra_rhino, spectra_fm, freq_rhino, freq_fm
    
    
    def run_acquisition(self):
        """
        Execute the main data acquisition loop for RHINO.
        
        This continuously collects spectra in the 60-85 MHz band and 
        transmits them to the logging server for storage and analysis.
        """
        if not self.pynq_available:
            print("[ERROR] Cannot run acquisition - PYNQ not available")
            return
        
        print("\n" + "="*70)
        print("STARTING RHINO DATA ACQUISITION")
        print("="*70)
        print(f"Observing Band: {RHINOConfig.BAND_START_MHZ}-{RHINOConfig.BAND_END_MHZ} MHz")
        print(f"Duration: {RHINOConfig.ACQUISITION_DURATION} seconds")
        print(f"Spectra per file: {RHINOConfig.SPECTRA_PER_FILE}")
        print(f"Target server: {RHINOConfig.SERVER_HOSTNAME}:{RHINOConfig.SERVER_PORT}")
        print("="*70 + "\n")
        
        self.acquisition_start_time = time.time()
        file_counter = 0
        
        try:
            while True:
                elapsed_time = time.time() - self.acquisition_start_time
                if elapsed_time >= RHINOConfig.ACQUISITION_DURATION:
                    print("\n[ACQUISITION] Target duration reached")
                    break
                
                print(f"\n{'─'*70}")
                print(f"File #{file_counter + 1}")
                print(f"Elapsed time: {elapsed_time:.1f} / {RHINOConfig.ACQUISITION_DURATION} seconds")
                print(f"{'─'*70}")
                
                # Collect batch of spectra (RHINO + FM bands)
                spectra_rhino, spectra_fm, freq_rhino, freq_fm = self.collect_spectra_batch(
                    RHINOConfig.SPECTRA_PER_FILE
                )
                
                # Calculate FM contamination statistics
                fm_mean = np.mean(spectra_fm)
                fm_std = np.std(spectra_fm)
                fm_max = np.max(spectra_fm)
                rhino_mean = np.mean(spectra_rhino)
                contamination_ratio = fm_mean / rhino_mean if rhino_mean > 0 else 0
                
                print(f"\n[FM CONTAMINATION]")
                print(f"    FM Band Power (mean): {fm_mean:.2e}")
                print(f"    FM Band Power (max): {fm_max:.2e}")
                print(f"    FM/RHINO Ratio: {contamination_ratio:.2e}")
                if contamination_ratio > 1e6:
                    print(f"    ⚠ WARNING: FM contamination is very high!")
                
                # Generate metadata
                timestamp = datetime.utcnow().isoformat()
                filename_rhino = f"rhino_science_{timestamp.replace(':', '-').replace('.', '_')}.npy"
                filename_fm = f"rhino_fm_{timestamp.replace(':', '-').replace('.', '_')}.npy"
                
                metadata = {
                    'filename_science': filename_rhino,
                    'filename_fm': filename_fm,
                    'timestamp': timestamp,
                    'experiment': 'RHINO',
                    'mode': 'direct_sampling',
                    'rhino_band_start_mhz': RHINOConfig.BAND_START_MHZ,
                    'rhino_band_end_mhz': RHINOConfig.BAND_END_MHZ,
                    'fm_band_start_mhz': RHINOConfig.FM_START_MHZ,
                    'fm_band_end_mhz': RHINOConfig.FM_END_MHZ,
                    'adc_sample_rate_hz': float(self.adc_sample_rate),
                    'target_sample_rate_mhz': RHINOConfig.TARGET_SAMPLE_RATE_MHZ,
                    'decimation_factor': RHINOConfig.DECIMATION_FACTOR,
                    'fft_size': RHINOConfig.FFT_SIZE,
                    'num_frames_averaged': RHINOConfig.NUMBER_OF_FRAMES,
                    'window_type': RHINOConfig.WINDOW_TYPE,
                    'num_spectra': RHINOConfig.SPECTRA_PER_FILE,
                    'spectrum_type': RHINOConfig.SPECTRUM_TYPE,
                    'rhino_frequency_bins': self.num_rhino_bins,
                    'fm_frequency_bins': self.num_fm_bins,
                    'frequency_resolution_khz': float((self.adc_sample_rate / RHINOConfig.FFT_SIZE) / 1e3),
                    'fm_contamination_mean': float(fm_mean),
                    'fm_contamination_max': float(fm_max),
                    'fm_rhino_ratio': float(contamination_ratio),
                }
                
                # Compute MD5 checksums for both bands
                if RHINOConfig.ENABLE_MD5_CHECKSUM:
                    md5_rhino = hashlib.md5(spectra_rhino).hexdigest()
                    md5_fm = hashlib.md5(spectra_fm).hexdigest()
                    metadata['md5sum_science'] = md5_rhino
                    metadata['md5sum_fm'] = md5_fm
                    print(f"[CHECKSUM] Science band MD5: {md5_rhino}")
                    print(f"[CHECKSUM] FM band MD5: {md5_fm}")
                
                # Optionally save local copy
                if RHINOConfig.SAVE_LOCAL_COPY:
                    import os
                    os.makedirs(RHINOConfig.LOCAL_DATA_DIR, exist_ok=True)
                    
                    # Save science band
                    np.save(os.path.join(RHINOConfig.LOCAL_DATA_DIR, filename_rhino), spectra_rhino)
                    np.save(os.path.join(RHINOConfig.LOCAL_DATA_DIR, 
                                        filename_rhino.replace('.npy', '_freq.npy')), freq_rhino)
                    
                    # Save FM band
                    np.save(os.path.join(RHINOConfig.LOCAL_DATA_DIR, filename_fm), spectra_fm)
                    np.save(os.path.join(RHINOConfig.LOCAL_DATA_DIR,
                                        filename_fm.replace('.npy', '_freq.npy')), freq_fm)
                    
                    print(f"[LOCAL STORAGE] Saved both bands to {RHINOConfig.LOCAL_DATA_DIR}")
                
                # Transmit science band to logging server
                # (Can also transmit FM band if needed for centralized monitoring)
                transmit_spectrum_data(spectra_rhino, metadata)
                
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
    
    
    def _print_statistics(self):
        """
        Display acquisition statistics and performance metrics.
        """
        print("\n" + "="*70)
        print("ACQUISITION STATISTICS")
        print("="*70)
        
        if self.acquisition_start_time is not None:
            total_time = time.time() - self.acquisition_start_time
            print(f"Total acquisition time: {total_time:.2f} seconds")
            print(f"Total spectra collected: {self.total_spectra_collected}")
            print(f"Total files transmitted: {self.total_files_transmitted}")
            
            if total_time > 0:
                avg_rate = self.total_spectra_collected / total_time
                print(f"Average acquisition rate: {avg_rate:.2f} spectra/second")
            
            # Data volume for both bands
            rhino_data_mb = (self.total_spectra_collected * 
                           self.num_rhino_bins * 4) / (1024*1024)  # 4 bytes per float32
            fm_data_mb = (self.total_spectra_collected * 
                        self.num_fm_bins * 4) / (1024*1024)
            total_data_mb = rhino_data_mb + fm_data_mb
            
            print(f"Total data volume:")
            print(f"  RHINO band (60-85 MHz): {rhino_data_mb:.2f} MB")
            print(f"  FM band (88-108 MHz): {fm_data_mb:.2f} MB")
            print(f"  Total: {total_data_mb:.2f} MB")
            
        print("="*70 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main entry point for the RHINO spectrum acquisition script.
    """
    print("\n" + "="*70)
    print("RHINO 21cm Global Signal Acquisition - Initializing")
    print("Direct Sampling Mode")
    print("="*70 + "\n")
    
    # Validate configuration
    print("[VALIDATION] Checking configuration parameters...")
    
    # Check decimation is 1
    if RHINOConfig.DECIMATION_FACTOR != 1:
        print(f"[WARNING] Decimation set to {RHINOConfig.DECIMATION_FACTOR}")
        print("          For direct sampling, decimation should be 1")
        print("          Proceeding anyway...")
    
    # Check FFT size is power of 2
    if not (RHINOConfig.FFT_SIZE & (RHINOConfig.FFT_SIZE - 1) == 0):
        print(f"[ERROR] FFT size must be a power of 2: {RHINOConfig.FFT_SIZE}")
        sys.exit(1)
    
    # Check number of frames
    if not (1 <= RHINOConfig.NUMBER_OF_FRAMES <= 64):
        print(f"[ERROR] Number of frames must be between 1 and 64")
        sys.exit(1)
    
    # Check band makes sense
    if RHINOConfig.BAND_START_MHZ >= RHINOConfig.BAND_END_MHZ:
        print(f"[ERROR] Invalid band: {RHINOConfig.BAND_START_MHZ}-{RHINOConfig.BAND_END_MHZ} MHz")
        sys.exit(1)
    
    print("    ✓ Configuration valid\n")
    
    # Initialize acquisition system
    try:
        acquisition_system = RHINODirectSampling()
    except Exception as e:
        print(f"\n[FATAL ERROR] Failed to initialize acquisition system: {e}")
        sys.exit(1)
    
    # Run acquisition
    try:
        acquisition_system.run_acquisition()
    except Exception as e:
        print(f"\n[FATAL ERROR] Acquisition failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n[SHUTDOWN] Acquisition complete - exiting gracefully")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()