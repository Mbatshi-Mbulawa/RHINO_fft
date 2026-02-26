
import numpy as np
import time
import sys
import hashlib
import json
from datetime import datetime
from websockets.sync.client import connect
from pickle import dumps
import itertools


# CONFIGURATION PARAMETERS FOR RHINO DIRECT SAMPLING

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
    SERVER_HOSTNAME = "localhost"  # IP address of logging computer
    SERVER_PORT = 8765                  # WebSocket server port
    WEBSOCKETS_MAX_SIZE = int(1e6)      # Maximum WebSocket message size (bytes)
    WEBSOCKETS_CHUNK_SIZE = int(1e6)    # Data chunk size for transmission (bytes)
      
    TARGET_SAMPLE_RATE_MHZ = 300.0  # Target ADC sample rate in MHz
    
    DECIMATION_FACTOR = 1  # NO DECIMATION - keep full sample rate
                           # Critical for maintaining bandwidth to 150 MHz
    
    FFT_SIZE = 16384       # Starting point (can increase to 32768 or 65536)
                           # Frequency resolution @ 300 MHz = 18.3 kHz per bin
    
    NUMBER_OF_FRAMES = 32  # Number of FFT frames to average (1-64)
                           # More frames = lower noise, but slower cadence
    
    WINDOW_TYPE = 'blackman'  # Window function to reduce spectral leakage
                             # Options: 'rectangular', 'hanning', 'hamming', 'blackman'
    
    SPECTRUM_TYPE = 'power'  # Output type: 'magnitude', 'power', 'log'
                            # 'power' is standard for radio astronomy (units: K^2)
    
    # RHINO Science Band
    # BAND_START_MHZ = 60.0    # Start of RHINO observing band
    # BAND_END_MHZ = 85.0      # End of RHINO observing band
    
    # FM Contamination Band (for monitoring)
    # FM_START_MHZ = 88.0      # Start of FM broadcast band
    # FM_END_MHZ = 108.0       # End of FM broadcast band

    # Time Averaging Configuration

    INTEGRATION_TIME_SECONDS = 1.0

    # Data Collection
    # --------------
    ACQUISITION_DURATION = 3600    # Total acquisition time in seconds (1 hour)
    SPECTRA_PER_FILE = 60          # Number of spectra per file (e.g., 1 per minute)
    
    # Calibration
    # ----------
    CALIBRATION_MODE = 1         # ADC calibration: 1 = Mode 1, 2 = Mode 2
    
    # Data Quality
    # -----------
    ENABLE_MD5_CHECKSUM = True   # Compute MD5 checksums for data integrity
    
    # Logging
    # ------
    VERBOSE = True               # Enable detailed console output
    SAVE_LOCAL_COPY = False      # Save data locally before transmission
    LOCAL_DATA_DIR = "./data"    # Directory for local data storage


# DATA TRANSMISSION UTILITIES
# ---------------------------
def send_array(websocket, arr, chunk_size=RHINOConfig.WEBSOCKETS_CHUNK_SIZE):
    """
    Transmit a numpy array over WebSocket connection in chunks.

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

# RHINO SPECTRUM ACQUISITION SYSTEM - DIRECT SAMPLING MODE
# --------------------------------------------------------

class RHINOAcquisition:
    """
    RHINO spectrum acquisition controller for direct sampling mode.
    
    """
    def __init__(self, overlay_path=None):

        try:
            import pynq
            from rfsoc_sam import overlay
            self.pynq_available = True
        except ImportError as e:
            print(f"[ERROR] Required PYNQ libraries not found: {e}")
            print("          This script must run on the RFSoC board with PYNQ installed")
            self.pynq_available = False
            return
        
        # Load thr spectrum analyzer overlay
        print("\n[INITIALIZATION] Loading FPGA overlay...")
        try:
            if overlay_path is None:
                self.ol = overlay.SpectrumAnalyzerOverlay()
            else:
                self.ol = pynq.Overlay(overlay_path)
            print("     Overlay loaded successfully")
        except Exception as e:
            print(f"   [ERROR] Failed to load overlay: {e}")
            raise

        # Get references to hardware components 
        self.receiver = self.ol.radio.receiver

        print("          Hardware components initialized successfully")

        # Configure for direct sampling
        self._configure_system() 

        # Initialize statistics
        self.total_spectra_collected = 0
        self.total_files_transmitted = 0
        self.acquisition_start_time = None

    def _configure_system(self):
        """
        Configure RFSoC for direct sampling at 300 MHz.
        """
        print("\n[CONFIGURATION] Setting up direct sampling...")
        
        # Set decimation to 1 (no decimation)
        print("\n    === Direct Sampling Configuration ===")
        print("    Mode: Direct RF sampling (no DDC, no decimation)")
        self.receiver.analyser.decimation_factor = RHINOConfig.DECIMATION_FACTOR
        print(f"    Decimation: {RHINOConfig.DECIMATION_FACTOR}× (NONE)")
        
        # Read actual sample rate
        self.adc_sample_rate = self.receiver.analyser.sample_frequency
        print(f"    ADC Sample Rate: {self.adc_sample_rate / 1e6:.2f} MHz")
        
        # Verify against target
        target_rate = RHINOConfig.TARGET_SAMPLE_RATE_MHZ * 1e6
        rate_error = abs(self.adc_sample_rate - target_rate) / target_rate
        
        if rate_error > 0.05:  # 5% tolerance
            print(f"\n    WARNING: Sample rate mismatch!")
            print(f"      Target: {RHINOConfig.TARGET_SAMPLE_RATE_MHZ} MHz")
            print(f"      Actual: {self.adc_sample_rate/1e6:.2f} MHz")
            print(f"      Error: {rate_error*100:.1f}%")
        else:
            print(f"    ✓ Sample rate matches target")
        
        # Calculate Nyquist
        nyquist_freq = (self.adc_sample_rate / 2) / 1e6
        print(f"    Nyquist Frequency: {nyquist_freq:.2f} MHz")
        print(f"    Coverage: DC to {nyquist_freq:.2f} MHz")
        
        # Set FFT size
        self.receiver.analyser.fftsize = RHINOConfig.FFT_SIZE
        freq_resolution = (self.adc_sample_rate / RHINOConfig.FFT_SIZE) / 1e3
        print(f"\n    FFT Size: {RHINOConfig.FFT_SIZE} points")
        print(f"    Frequency Resolution: {freq_resolution:.3f} kHz per bin")
        
        # Set hardware averaging
        self.receiver.analyser.number_frames = RHINOConfig.NUMBER_OF_FRAMES
        
        # Calculate single spectrum time
        self.spectrum_time = (RHINOConfig.NUMBER_OF_FRAMES * 
                             RHINOConfig.FFT_SIZE / self.adc_sample_rate)
        print(f"\n    Hardware Frame Averaging: {RHINOConfig.NUMBER_OF_FRAMES} frames")
        print(f"    Time per spectrum: {self.spectrum_time*1e3:.2f} ms")
        
        # Calculate how many spectra to average for integration time
        self.spectra_per_integration = max(1, int(
            RHINOConfig.INTEGRATION_TIME_SECONDS / self.spectrum_time
        ))
        actual_integration = self.spectra_per_integration * self.spectrum_time
        print(f"\n    === Time Averaging Configuration ===")
        print(f"    Target Integration: {RHINOConfig.INTEGRATION_TIME_SECONDS} seconds")
        print(f"    Spectra to Average: {self.spectra_per_integration}")
        print(f"    Actual Integration: {actual_integration:.3f} seconds")
        
        # Set window function
        self.receiver.analyser.window = RHINOConfig.WINDOW_TYPE
        print(f"\n    Window Function: {RHINOConfig.WINDOW_TYPE}")
        
        # Set spectrum type (power = |FFT|^2)
        self.receiver.analyser.spectrum_type = RHINOConfig.SPECTRUM_TYPE
        print(f"    Spectrum Type: {RHINOConfig.SPECTRUM_TYPE} (|FFT|²)")
        
        # Set calibration mode
        self.receiver.analyser.calibration_mode = RHINOConfig.CALIBRATION_MODE
        print(f"    Calibration Mode: {RHINOConfig.CALIBRATION_MODE}")
        
        # Create frequency axis
        self.frequency_axis = np.arange(RHINOConfig.FFT_SIZE) * (
            self.adc_sample_rate / RHINOConfig.FFT_SIZE
        ) / 1e6  # MHz
        
        print("\n    ✓ Configuration complete\n")

    def capture_single_spectrum(self):
        """
        Capture a single hardware-averaged spectrum.
        Triggers the complete hardware pipeline:

        Tells the DMA to start capturing from the FFT output
        Waits for 32 FFT frames to be computed and averaged (1.75 ms)
        DMA transfers the result from FPGA memory to CPU-accessible RAM
        Returns the data as a numpy array of shape (FFT_SIZE,)
        
        Returns:
        -------
        spectrum : np.ndarray
            Power spectrum (|FFT|²), length = FFT_SIZE
            Covers DC to Nyquist (0 to 150 MHz at 300 MHz sampling)
        """
        # This captures one spectrum with hardware averaging
        # (NUMBER_OF_FRAMES averaged in FPGA)
        spectrum = self.receiver.analyser.spectrum_data()
        return spectrum
    
    def capture_time_averaged_spectrum(self):
        """
        Capture and time-average multiple spectra.
        
        
        Returns:
        -------
        averaged_spectrum : np.ndarray
            Time-averaged power spectrum over integration time
        """
        # Accumulator for averaging
        accumulated = np.zeros(RHINOConfig.FFT_SIZE, dtype=np.float64)
        
        # Collect and average spectra
        for i in range(self.spectra_per_integration):
            spectrum = self.capture_single_spectrum()
            accumulated += spectrum
            self.total_spectra_collected += 1
        
        # Compute mean
        averaged_spectrum = accumulated / self.spectra_per_integration
        
        return averaged_spectrum.astype(np.float32)
    
    def collect_spectrum_batch(self, num_averaged_spectra):
        """
        Collect multiple time-averaged spectra.
        
        Parameters:
        ----------
        num_averaged_spectra : int
            Number of time-averaged spectra to collect
        
        Returns:
        -------
        spectra_batch : np.ndarray
            2D array of shape (num_averaged_spectra, FFT_SIZE)
            Each row is one time-averaged spectrum
        """
        print(f"[ACQUISITION] Collecting {num_averaged_spectra} time-averaged spectra...")
        print(f"    Integration time per spectrum: {self.spectra_per_integration * self.spectrum_time:.3f} sec")
        
        # Pre-allocate
        spectra_batch = np.zeros((num_averaged_spectra, RHINOConfig.FFT_SIZE), 
                                 dtype=np.float32)
        
        batch_start = time.time()
        
        # Collect time-averaged spectra
        for i in range(num_averaged_spectra):
            spectra_batch[i, :] = self.capture_time_averaged_spectrum()
            
            if RHINOConfig.VERBOSE and (i+1) % 10 == 0:
                elapsed = time.time() - batch_start
                rate = (i+1) / elapsed
                eta = (num_averaged_spectra - i - 1) / rate if rate > 0 else 0
                print(f"    Progress: {i+1}/{num_averaged_spectra} "
                      f"({rate:.2f} avg_spec/sec, ETA: {eta:.1f}s)")
        
        batch_duration = time.time() - batch_start
        
        print(f"    ✓ Batch complete: {batch_duration:.2f} seconds")
        print(f"    Rate: {num_averaged_spectra/batch_duration:.2f} averaged spectra/sec")
        
        return spectra_batch
    
    def run_acquisition(self):
        """
        Main acquisition loop.
        
        Continuously collects time-averaged spectra and transmits to server.
        """
        if not self.pynq_available:
            print("[ERROR] Cannot run - PYNQ not available")
            return
        
        print("\n" + "="*70)
        print("STARTING DATA ACQUISITION")
        print("="*70)
        print(f"Duration: {RHINOConfig.ACQUISITION_DURATION} seconds")
        print(f"Integration time: {self.spectra_per_integration * self.spectrum_time:.3f} sec")
        print(f"Averaged spectra per file: {RHINOConfig.SPECTRA_PER_FILE}")
        print(f"Server: {RHINOConfig.SERVER_HOSTNAME}:{RHINOConfig.SERVER_PORT}")
        print("="*70 + "\n")
        
        self.acquisition_start_time = time.time()
        file_counter = 0
        
        try:
            while True:
                # Check duration
                elapsed = time.time() - self.acquisition_start_time
                if elapsed >= RHINOConfig.ACQUISITION_DURATION:
                    print("\n[ACQUISITION] Target duration reached")
                    break
                
                print(f"\n{'─'*70}")
                print(f"File #{file_counter + 1}")
                print(f"Elapsed: {elapsed:.1f} / {RHINOConfig.ACQUISITION_DURATION} sec")
                print(f"{'─'*70}")
                
                # Collect batch of time-averaged spectra
                spectra_batch = self.collect_spectrum_batch(RHINOConfig.SPECTRA_PER_FILE)
                
                # Generate metadata
                #timestamp = datetime.utcnow().isoformat()
                #filename = f"rhino_spectrum_{timestamp.replace(':', '-').replace('.', '_')}.npy"
                
                #metadata = {
                 #   'filename': filename,
                  #  'timestamp': timestamp,
                   # 'experiment': 'RHINO',
                    #'mode': 'direct_sampling_time_averaged',
                    #'adc_sample_rate_hz': float(self.adc_sample_rate),
                    #'target_sample_rate_mhz': RHINOConfig.TARGET_SAMPLE_RATE_MHZ,
                    #'nyquist_frequency_mhz': float(self.adc_sample_rate / 2e6),
                    #'decimation_factor': RHINOConfig.DECIMATION_FACTOR,
                    #'fft_size': RHINOConfig.FFT_SIZE,
                    #'frequency_resolution_khz': float((self.adc_sample_rate / RHINOConfig.FFT_SIZE) / 1e3),
                    #'hardware_frames_averaged': RHINOConfig.NUMBER_OF_FRAMES,
                    #'spectra_per_time_average': self.spectra_per_integration,
                    #'integration_time_seconds': float(self.spectra_per_integration * self.spectrum_time),
                    #'window_type': RHINOConfig.WINDOW_TYPE,
                    #'spectrum_type': RHINOConfig.SPECTRUM_TYPE,
                    #'num_averaged_spectra': RHINOConfig.SPECTRA_PER_FILE,
                    #'total_raw_spectra_collected': int(self.total_spectra_collected),
                    #'frequency_coverage': f"DC to {self.adc_sample_rate/2e6:.1f} MHz",
                #}
                
                # Compute checksum
                if RHINOConfig.ENABLE_MD5_CHECKSUM:
                    md5sum = hashlib.md5(spectra_batch).hexdigest()
                    #metadata['md5sum'] = md5sum
                    print(f"[CHECKSUM] MD5: {md5sum}")
                
                # Save locally if requested
                if RHINOConfig.SAVE_LOCAL_COPY:
                    import os
                    os.makedirs(RHINOConfig.LOCAL_DATA_DIR, exist_ok=True)
                    
                    # Save spectra
                    #local_path = os.path.join(RHINOConfig.LOCAL_DATA_DIR, filename)
                    np.save(local_path, spectra_batch)
                    
                    # Save frequency axis
                    #freq_file = filename.replace('.npy', '_freq.npy')
                    #np.save(os.path.join(RHINOConfig.LOCAL_DATA_DIR, freq_file), 
                          # self.frequency_axis)
                    
                    #print(f"[LOCAL] Saved to {local_path}")
                
                # Transmit to server
                #transmit_spectrum_data(spectra_batch, metadata)
                
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
        """Print acquisition statistics."""
        print("\n" + "="*70)
        print("ACQUISITION STATISTICS")
        print("="*70)
        
        if self.acquisition_start_time is not None:
            total_time = time.time() - self.acquisition_start_time
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Raw spectra collected: {self.total_spectra_collected}")
            print(f"Averaged spectra: {self.total_files_transmitted * RHINOConfig.SPECTRA_PER_FILE}")
            print(f"Files transmitted: {self.total_files_transmitted}")
            
            # Data volume
            data_mb = (self.total_files_transmitted * RHINOConfig.SPECTRA_PER_FILE * 
                      RHINOConfig.FFT_SIZE * 4) / (1024*1024)
            print(f"Total data volume: {data_mb:.2f} MB")
            
        print("="*70 + "\n")

# MAIN EXECUTION
def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("RHINO Acquisition System - Initializing")
    print("="*70 + "\n")
    
    # Validate configuration
    print("[VALIDATION] Checking configuration...")
    
    if RHINOConfig.DECIMATION_FACTOR != 1:
        print(f"[WARNING] Decimation = {RHINOConfig.DECIMATION_FACTOR}")
        print("          Should be 1 for direct sampling")
    
    if not (RHINOConfig.FFT_SIZE & (RHINOConfig.FFT_SIZE - 1) == 0):
        print(f"[ERROR] FFT size must be power of 2: {RHINOConfig.FFT_SIZE}")
        sys.exit(1)
    
    if not (1 <= RHINOConfig.NUMBER_OF_FRAMES <= 64):
        print(f"[ERROR] Number of frames must be 1-64")
        sys.exit(1)
    
    print("    ✓ Configuration valid\n")
    
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
    
    print("\n[SHUTDOWN] Acquisition complete")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()