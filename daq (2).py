#!/usr/bin/env python3
"""
RHINO 21cm Global Signal Acquisition - Simplified Direct Sampling
================================================================

Author: Mbatshi Jerry Junior Mbulawa
Project: RFSoC DSP and PYNQ - RHINO Receiver Platform
Supervisor: Dr. Bull
Institution: University of Manchester (Jodrell Bank Centre for Astrophysics)
Date: February 2026

Description:
-----------
Simplified acquisition script for RHINO experiment based on these
specifications:
1. Direct sampling (no DDC, no decimation)
2. Single spectrum output (FFT squared for spectral power)
3. Time-averaged spectra (1 second integration)
4. No separate FM band extraction (handle in post-processing)

Current status:
--------------
Transmission is COMMENTED OUT while the acquisition pipeline is verified.
Search for "TO UNCOMMENT" to find every line to re-enable when ready.

References:
----------
[1] Bull et al. (2024), "RHINO: A large horn antenna for detecting the 21cm
    global signal", arXiv:2410.00076
"""

import numpy as np
import time
import sys
import hashlib
import json
from datetime import datetime

# ── TO UNCOMMENT when transmission is ready ──────────────────────────────────
# from websockets.sync.client import connect
# from pickle import dumps
# import itertools
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

class RHINOConfig:
    """
    Configuration for RHINO direct sampling acquisition.

    Based on PhD student specifications:
    - Direct sampling (no DDC)
    - 8192 point FFT (overlay maximum; 16384 requires custom Vivado overlay)
    - Time-averaged spectra
    - Spectral power output (FFT squared)
    """

    # =========================================================================
    # Network Configuration
    # (not active until transmission is re-enabled)
    # =========================================================================
    SERVER_HOSTNAME       = "192.168.1.100"  # IP address of logging computer
    SERVER_PORT           = 8765             # WebSocket server port
    WEBSOCKETS_MAX_SIZE   = int(1e6)         # Maximum message size (bytes)
    WEBSOCKETS_CHUNK_SIZE = int(1e6)         # Chunk size for transmission

    # =========================================================================
    # RF Configuration - DIRECT SAMPLING
    # =========================================================================
    TARGET_SAMPLE_RATE_MHZ = 4915.2  # Hardware-reported ADC rate.
                                      # The board runs at 4915.2 MHz, not the
                                      # 300 MHz originally assumed.
    DECIMATION_FACTOR = 1             # NO DECIMATION (direct sampling)

    # =========================================================================
    # FFT Configuration
    # =========================================================================
    FFT_SIZE       = 8192    # Maximum supported by the rfsoc_sam overlay.
                             # Valid options: 64, 128, 256, 512, 1024, 2048, 4096, 8192
                             # 16384 is NOT supported — requires a custom Vivado overlay.
                             # At 4915.2 MHz: resolution = 600 kHz per bin
                             # Covers DC to 2457.6 MHz (Nyquist)
    NUMBER_OF_FRAMES = 32    # Reference value for spectrum time calculation
    WINDOW_TYPE    = 'blackman'# You can have your choice of either 'Hanning','Blackman','Hamming', etc. 
                               # Based on the RHINO horn antenna specifications
    SPECTRUM_TYPE  = 'power'  # |FFT|^2

    # =========================================================================
    # Time Averaging Configuration
    # =========================================================================
    # Set to 0.0 to capture exactly ONE frame per output spectrum.
    # This is the safest starting point — confirms get_frame() returns before
    # asking it to loop thousands of times.
    # Once get_frame() is confirmed working, restore to 1.0 for proper
    # 1-second time averaging.
    INTEGRATION_TIME_SECONDS = 0.0  # 0.0 = single frame only (test mode)

    # =========================================================================
    # Data Collection
    # =========================================================================
    ACQUISITION_DURATION = 10   # Total run time in seconds.
                                 # 10 s is enough for a quick test.
                                 # Restore to 3600 for a real observation.
    SPECTRA_PER_FILE = 5         # Spectra per batch before transmission fires.
                                 # Kept low for testing.
                                 # Restore to 60 for a real observation.

    # =========================================================================
    # Data Quality
    # =========================================================================
    ENABLE_MD5_CHECKSUM = True   # Checksum computed when transmitting

    # =========================================================================
    # Logging
    # =========================================================================
    VERBOSE = True

    # =========================================================================
    # Local save
    # =========================================================================
    SAVE_LOCAL_COPY = False      # Set True to save .npy files locally
    LOCAL_DATA_DIR  = "./data"


# =============================================================================
# DATA TRANSMISSION UTILITIES
# (fully commented out — uncomment when ready to transmit)
# =============================================================================

# ── TO UNCOMMENT when transmission is ready ──────────────────────────────────
# def send_array(websocket, arr, chunk_size=RHINOConfig.WEBSOCKETS_CHUNK_SIZE):
#     """Transmit a numpy array over WebSocket in chunks."""
#     bytes_arr = dumps(arr)
#     if RHINOConfig.VERBOSE:
#         print(f"    Serialized array size: {len(bytes_arr) / (1024*1024):.2f} MB")
#     t_start = time.time()
#     chunk_count = 0
#     for chunk in itertools.batched(bytes_arr, chunk_size):
#         websocket.send(bytes(chunk), text=False)
#         chunk_count += 1
#     transmission_time = time.time() - t_start
#     if RHINOConfig.VERBOSE:
#         print(f"    Transmitted {chunk_count} chunks in {transmission_time:.3f} s")
#         print(f"    Throughput: {(len(bytes_arr)/(1024*1024))/transmission_time:.2f} MB/s")
#
#
# def transmit_spectrum_data(spectrum_data, metadata):
#     """Transmit spectrum data with metadata to logging server."""
#     server_url = f"ws://{RHINOConfig.SERVER_HOSTNAME}:{RHINOConfig.SERVER_PORT}"
#     try:
#         if RHINOConfig.VERBOSE:
#             print(f"\n[TRANSMISSION] Connecting to {server_url}")
#         with connect(server_url, max_size=RHINOConfig.WEBSOCKETS_MAX_SIZE) as websocket:
#             if RHINOConfig.VERBOSE:
#                 print("    Sending metadata...")
#             websocket.send(json.dumps(metadata), text=True)
#             if RHINOConfig.VERBOSE:
#                 print("    Sending spectrum data...")
#             send_array(websocket, spectrum_data)
#             print(f"[TRANSMISSION] Successfully transmitted: {metadata['filename']}")
#     except ConnectionRefusedError:
#         print(f"[ERROR] Cannot connect to server at {server_url}")
#         print("        Ensure rhino_data_server.py is running on logging computer")
#         raise
#     except Exception as e:
#         print(f"[ERROR] Transmission failed: {e}")
#         raise
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# RHINO SPECTRUM ACQUISITION SYSTEM
# =============================================================================

class RHINOAcquisition:
    """
    RHINO spectrum acquisition controller for direct sampling mode.
    """

    def __init__(self, overlay_path=None):

        print("=" * 70)
        print("RHINO 21cm Global Signal Acquisition System")
        print("Simplified Direct Sampling Mode")
        print("Jodrell Bank Centre for Astrophysics - University of Manchester")
        print("=" * 70)

        # Import PYNQ and rfsoc_sam
        try:
            import pynq
            from rfsoc_sam import overlay
            self.pynq_available = True
        except ImportError as e:
            print(f"[ERROR] PYNQ libraries not found: {e}")
            print("        This must run on the RFSoC board with PYNQ installed")
            self.pynq_available = False
            return

        # Load the FPGA overlay
        print("\n[INITIALIZATION] Loading FPGA overlay...")
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
        # Confirmed by dir() inspection on the board:
        #   ol.radio.receiver.channel_00.spectrum_analyser
        # Four channels exist (00, 02, 20, 22) — we use channel_00.
        self.receiver = self.ol.radio.receiver
        self.analyser = self.ol.radio.receiver.channel_00.spectrum_analyser
        print("    ✓ Hardware initialized")

        # Configure hardware
        self._configure_system()

        # Statistics counters
        self.total_spectra_collected = 0
        self.total_files_transmitted = 0
        self.acquisition_start_time  = None


    def _configure_system(self):
        """Configure RFSoC for direct sampling."""

        print("\n[CONFIGURATION] Setting up direct sampling...")
        print("\n    === Direct Sampling Configuration ===")
        print("    Mode: Direct RF sampling (no DDC, no decimation)")

        # Decimation = 1 (no frequency mixing or rate reduction)
        self.analyser.decimation_factor = RHINOConfig.DECIMATION_FACTOR
        print(f"    Decimation: {RHINOConfig.DECIMATION_FACTOR}x (NONE)")

        # Read the actual ADC sample rate from hardware
        self.adc_sample_rate = self.analyser.sample_frequency
        print(f"    ADC Sample Rate: {self.adc_sample_rate / 1e6:.2f} MHz")

        # Warn if hardware rate differs from expected
        target_rate = RHINOConfig.TARGET_SAMPLE_RATE_MHZ * 1e6
        rate_error  = abs(self.adc_sample_rate - target_rate) / target_rate
        if rate_error > 0.05:
            print(f"\n    WARNING: Sample rate mismatch!")
            print(f"      Target: {RHINOConfig.TARGET_SAMPLE_RATE_MHZ} MHz")
            print(f"      Actual: {self.adc_sample_rate / 1e6:.2f} MHz")
            print(f"      Error:  {rate_error * 100:.1f}%")
        else:
            print(f"    ✓ Sample rate matches target")

        # Nyquist and coverage
        nyquist_freq = (self.adc_sample_rate / 2) / 1e6
        print(f"    Nyquist Frequency: {nyquist_freq:.2f} MHz")
        print(f"    Coverage: DC to {nyquist_freq:.2f} MHz")

        # FFT size
        self.analyser.fft_size = RHINOConfig.FFT_SIZE
        freq_resolution = (self.adc_sample_rate / RHINOConfig.FFT_SIZE) / 1e3
        print(f"\n    FFT Size: {RHINOConfig.FFT_SIZE} points")
        print(f"    Frequency Resolution: {freq_resolution:.3f} kHz per bin")

        # number_frames is not settable on this overlay version.
        # spectrum_time used only to estimate integration loop count.
        self.spectrum_time = (RHINOConfig.NUMBER_OF_FRAMES *
                              RHINOConfig.FFT_SIZE / self.adc_sample_rate)
        print(f"\n    Hardware Frame Averaging: {RHINOConfig.NUMBER_OF_FRAMES} frames")
        print(f"    Estimated time per spectrum: {self.spectrum_time * 1e3:.2f} ms")

        # Frames per output spectrum
        print(f"\n    === Time Averaging Configuration ===")
        if RHINOConfig.INTEGRATION_TIME_SECONDS == 0.0:
            self.spectra_per_integration = 1
            print("    Integration: 0.0 s (single frame per output — test mode)")
        else:
            self.spectra_per_integration = max(1, int(
                RHINOConfig.INTEGRATION_TIME_SECONDS / self.spectrum_time
            ))
            actual_integration = self.spectra_per_integration * self.spectrum_time
            print(f"    Target Integration: {RHINOConfig.INTEGRATION_TIME_SECONDS} s")
            print(f"    Spectra to Average: {self.spectra_per_integration}")
            print(f"    Actual Integration: {actual_integration:.3f} s")

        # Window function
        self.analyser.window = RHINOConfig.WINDOW_TYPE
        print(f"\n    Window Function: {RHINOConfig.WINDOW_TYPE}")

        # Spectrum type: power = |FFT|^2
        self.analyser.spectrum_type = RHINOConfig.SPECTRUM_TYPE
        print(f"    Spectrum Type: {RHINOConfig.SPECTRUM_TYPE} (|FFT|^2)")

        # calibration_mode does not exist on this overlay version — removed.

        # Enable DMA — without this get_frame() blocks forever
        self.analyser.dma_enable = 1
        print("    DMA: enabled")

        # Frequency axis in MHz
        self.frequency_axis = np.arange(RHINOConfig.FFT_SIZE) * (
            self.adc_sample_rate / RHINOConfig.FFT_SIZE
        ) / 1e6

        print("\n    ✓ Configuration complete\n")


    def capture_single_spectrum(self):
        """
        Capture one hardware-averaged spectrum from the FPGA via DMA.

        Returns
        -------
        spectrum : np.ndarray  shape (FFT_SIZE,)  dtype float32
        """
        print("    [DEBUG] Calling get_frame()...")
        spectrum = self.analyser.get_frame()
        print(f"    [DEBUG] get_frame() returned: shape={spectrum.shape}, "
              f"dtype={spectrum.dtype}, "
              f"min={spectrum.min():.3e}, max={spectrum.max():.3e}")
        return spectrum


    def capture_time_averaged_spectrum(self):
        """
        Capture and average spectra over the integration period.

        Returns
        -------
        averaged_spectrum : np.ndarray  shape (FFT_SIZE,)  dtype float32
        """
        accumulated = np.zeros(RHINOConfig.FFT_SIZE, dtype=np.float64)

        for i in range(self.spectra_per_integration):
            accumulated += self.capture_single_spectrum()
            self.total_spectra_collected += 1

        return (accumulated / self.spectra_per_integration).astype(np.float32)


    def collect_spectrum_batch(self, num_averaged_spectra):
        """
        Collect a batch of time-averaged spectra.

        Parameters
        ----------
        num_averaged_spectra : int

        Returns
        -------
        spectra_batch : np.ndarray  shape (num_averaged_spectra, FFT_SIZE)
        """
        print(f"[ACQUISITION] Collecting {num_averaged_spectra} "
              f"time-averaged spectra...")
        print(f"    Integration time per spectrum: "
              f"{self.spectra_per_integration * self.spectrum_time:.3f} sec")

        spectra_batch = np.zeros(
            (num_averaged_spectra, RHINOConfig.FFT_SIZE), dtype=np.float32
        )
        batch_start = time.time()

        for i in range(num_averaged_spectra):
            print(f"    Attempting to read spectrum {i+1}/{num_averaged_spectra}...")
            spectra_batch[i, :] = self.capture_time_averaged_spectrum()
            print(f"    ✓ Spectrum {i+1} acquired, shape: {spectra_batch[i].shape}")

            if RHINOConfig.VERBOSE and (i + 1) % 10 == 0:
                elapsed = time.time() - batch_start
                rate    = (i + 1) / elapsed
                eta     = (num_averaged_spectra - i - 1) / rate if rate > 0 else 0
                print(f"    Progress: {i+1}/{num_averaged_spectra} "
                      f"({rate:.2f} avg_spec/sec, ETA: {eta:.1f}s)")

        batch_duration = time.time() - batch_start
        print(f"    ✓ Batch complete: {batch_duration:.2f} seconds")
        return spectra_batch


    def run_acquisition(self):
        """Main acquisition loop."""

        if not self.pynq_available:
            print("[ERROR] Cannot run — PYNQ not available")
            return

        print("\n" + "=" * 70)
        print("STARTING DATA ACQUISITION")
        print("=" * 70)
        print(f"Duration:          {RHINOConfig.ACQUISITION_DURATION} seconds")
        print(f"Integration time:  "
              f"{self.spectra_per_integration * self.spectrum_time:.3f} sec")
        print(f"Spectra per batch: {RHINOConfig.SPECTRA_PER_FILE}")
        # ── TO UNCOMMENT when transmission is ready ───────────────────────────
        # print(f"Server: {RHINOConfig.SERVER_HOSTNAME}:{RHINOConfig.SERVER_PORT}")
        # ─────────────────────────────────────────────────────────────────────
        print("=" * 70 + "\n")

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
                print(f"Elapsed: {elapsed:.1f} / "
                      f"{RHINOConfig.ACQUISITION_DURATION} sec")
                print(f"{'─' * 70}")

                # ── Collect spectra ───────────────────────────────────────────
                spectra_batch = self.collect_spectrum_batch(
                    RHINOConfig.SPECTRA_PER_FILE
                )

                # ── Print spectrum for visual inspection ──────────────────────
                print(f"\n[SPECTRUM DATA] Batch shape: {spectra_batch.shape}")
                print(f"[SPECTRUM DATA] First spectrum (first 10 bins): "
                      f"{spectra_batch[0, :10]}")
                print(f"[SPECTRUM DATA] Min:  {spectra_batch[0].min():.4e}  "
                      f"Max: {spectra_batch[0].max():.4e}  "
                      f"Mean: {spectra_batch[0].mean():.4e}")

                # ── ASCII spectrum plot ───────────────────────────────────────
                # Downsample to 60 columns so it fits a terminal window.
                # Values are in dBFS (negative numbers are normal for noise).
                spec     = spectra_batch[0]
                n_cols   = 60
                step     = max(1, len(spec) // n_cols)
                ds_spec  = np.array([spec[i:i+step].mean()
                                     for i in range(0, len(spec)-step+1, step)])
                ds_freq  = self.frequency_axis[::step][:len(ds_spec)]

                # Convert to dB (values from get_frame() are already in dBFS)
                db_vals  = ds_spec  # already dB scale
                db_min   = db_vals.min()
                db_max   = db_vals.max()
                db_range = db_max - db_min if db_max != db_min else 1.0

                bar_height = 10   # rows tall
                print(f"\n[SPECTRUM PLOT] {db_min:.1f} dBFS (bottom) "
                      f"to {db_max:.1f} dBFS (top)")
                print(f"                0 MHz (left) to "
                      f"{ds_freq[-1]:.0f} MHz (right)\n")

                for row in range(bar_height, 0, -1):
                    threshold = db_min + (row / bar_height) * db_range
                    line = "".join(
                        "█" if v >= threshold else " " for v in db_vals
                    )
                    label = f"{db_min + (row/bar_height)*db_range:7.1f} |"
                    print(f"  {label}{line}")

                print("         +" + "─" * len(db_vals))
                print(f"          0{'MHz':>{len(db_vals)//2}}"
                      f"{'Nyquist':>{len(db_vals)//2}}\n")

                # ── TO UNCOMMENT when transmission is ready ───────────────────
                #
                # Step 1: Generate metadata
                # timestamp = datetime.utcnow().isoformat()
                # filename  = (f"rhino_spectrum_"
                #     f"{timestamp.replace(':', '-').replace('.', '_')}.npy")
                # metadata = {
                #     'filename':                 filename,
                #     'timestamp':                timestamp,
                #     'experiment':               'RHINO',
                #     'mode':                     'direct_sampling_time_averaged',
                #     'adc_sample_rate_hz':       float(self.adc_sample_rate),
                #     'target_sample_rate_mhz':   RHINOConfig.TARGET_SAMPLE_RATE_MHZ,
                #     'nyquist_frequency_mhz':    float(self.adc_sample_rate / 2e6),
                #     'decimation_factor':        RHINOConfig.DECIMATION_FACTOR,
                #     'fft_size':                 RHINOConfig.FFT_SIZE,
                #     'frequency_resolution_khz': float(
                #         (self.adc_sample_rate / RHINOConfig.FFT_SIZE) / 1e3),
                #     'hardware_frames_averaged': RHINOConfig.NUMBER_OF_FRAMES,
                #     'spectra_per_time_average': self.spectra_per_integration,
                #     'integration_time_seconds': float(
                #         self.spectra_per_integration * self.spectrum_time),
                #     'window_type':              RHINOConfig.WINDOW_TYPE,
                #     'spectrum_type':            RHINOConfig.SPECTRUM_TYPE,
                #     'num_averaged_spectra':     RHINOConfig.SPECTRA_PER_FILE,
                #     'total_raw_spectra':        int(self.total_spectra_collected),
                #     'frequency_coverage':       (f"DC to "
                #         f"{self.adc_sample_rate/2e6:.1f} MHz"),
                # }
                #
                # Step 2: Compute MD5 checksum
                # if RHINOConfig.ENABLE_MD5_CHECKSUM:
                #     md5sum = hashlib.md5(spectra_batch).hexdigest()
                #     metadata['md5sum'] = md5sum
                #     print(f"[CHECKSUM] MD5: {md5sum}")
                #
                # Step 3: Save locally (optional)
                # if RHINOConfig.SAVE_LOCAL_COPY:
                #     import os
                #     os.makedirs(RHINOConfig.LOCAL_DATA_DIR, exist_ok=True)
                #     local_path = os.path.join(RHINOConfig.LOCAL_DATA_DIR, filename)
                #     np.save(local_path, spectra_batch)
                #     freq_file = filename.replace('.npy', '_freq.npy')
                #     np.save(os.path.join(RHINOConfig.LOCAL_DATA_DIR, freq_file),
                #             self.frequency_axis)
                #     print(f"[LOCAL] Saved to {local_path}")
                #
                # Step 4: Transmit to logging server
                # transmit_spectrum_data(spectra_batch, metadata)
                #
                # ─────────────────────────────────────────────────────────────

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
        print("\n" + "=" * 70)
        print("ACQUISITION STATISTICS")
        print("=" * 70)
        if self.acquisition_start_time is not None:
            total_time = time.time() - self.acquisition_start_time
            print(f"Total time:            {total_time:.2f} seconds")
            print(f"Raw spectra collected: {self.total_spectra_collected}")
            print(f"Batches completed:     {self.total_files_transmitted}")
            data_mb = (self.total_files_transmitted *
                       RHINOConfig.SPECTRA_PER_FILE *
                       RHINOConfig.FFT_SIZE * 4) / (1024 * 1024)
            print(f"Data volume:           {data_mb:.2f} MB")
        print("=" * 70 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():

    print("\n" + "=" * 70)
    print("RHINO Acquisition System - Initializing")
    print("=" * 70 + "\n")

    print("[VALIDATION] Checking configuration...")

    if RHINOConfig.DECIMATION_FACTOR != 1:
        print(f"[WARNING] Decimation = {RHINOConfig.DECIMATION_FACTOR} "
              f"(should be 1 for direct sampling)")

    if not (RHINOConfig.FFT_SIZE & (RHINOConfig.FFT_SIZE - 1) == 0):
        print(f"[ERROR] FFT_SIZE must be a power of 2, got: {RHINOConfig.FFT_SIZE}")
        print(f"        Hardware confirmed value is 2048.")
        sys.exit(1)

    if not (1 <= RHINOConfig.NUMBER_OF_FRAMES <= 64):
        print("[ERROR] NUMBER_OF_FRAMES must be between 1 and 64")
        sys.exit(1)

    print("    ✓ Configuration valid\n")

    try:
        acquisition = RHINOAcquisition()
    except Exception as e:
        print(f"\n[FATAL] Initialization failed: {e}")
        sys.exit(1)

    try:
        acquisition.run_acquisition()
    except Exception as e:
        print(f"\n[FATAL] Acquisition failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n[SHUTDOWN] Acquisition complete")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
