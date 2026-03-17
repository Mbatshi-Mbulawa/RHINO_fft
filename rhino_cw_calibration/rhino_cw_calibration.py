import numpy as np
import matplotlib.pyplot as plt
import time
import os
import csv

# We wrap the rfsoc_sam  import in a try/except so the script can also be
# run in a "dry run" simulation mode on a PC without the board connected.
try:
    from rfsoc_sam import Overlay
    BOARD_CONNECTED = True
    print("[INFO] rfsoc_sam imported successfully. Board mode active.")
except ImportError:
    BOARD_CONNECTED = False
    print("[WARNING] rfsoc_sam not found. Running in SIMULATION mode.")
    print("         All ADC readings will be synthetic. For testing only.")

# =============================================================================
# CONFIGURATION - edit these values to change the experiment parameters
# =============================================================================

# The ADC DDC NCO is permanently fixed at this frequency (MHz) by the
# rfsoc-sam bitsteam. This is a hardware constant- do not change it.
DDC_NCO_MHZ = 1228.8

# ADC hardware sample rate in Hz when using rfsoc_sam
# rfsoc_sam does NOT decimate - the ADC tile runs at its full hardware rate
# of 4915.2 MSPS. The DDC shifts the band of interest down in frequency
# but the output sample rate remains at 4915.2 MSPS.
# NOTE: 16x decimation to 200 MSPS is a feature of the Custom system-overlay
# bitstream (not yet deployed) Not rfsoc_sam
ADC_FS_HZ = 4915.2e6 # 4915.2 MSPS - full hardware rate under rfsoc_sam

# FFT length used by rfsoc_sam internally (8192 pts confirmed in Session 1 plots)
# Bin spacing = 4915.2e6 / 8192 = ~ 599.6 kHz under rfsoc_sam
# ( Under the custom system_overlay with 16x decimation : 200e6 / 4096 = ~48.8 kHz)
N_FFT = 8192

# CW sweep range (MHz) 
SWEEP_START_MHZ = 50.0
SWEEP_STOP_MHZ = 200.0

# Step size for the frequency sweep.
# Bin spacing at 4915.2 MSPS / 8192 pts = ~ 599.6 kHz under rfsoc_sam.
BIN_SPACING_MHZ = (ADC_FS_HZ / N_FFT) / 1e6  # ~ 0.5996 MHz

# For a faster sweep during initial testing, we will use a coarser step (e.g 1 MHz)
# Change this to BIN_SPACING_MHZ for the final high-resolution calibration.
SWEEP_STEP_MHZ = 1.0 # coarse sweep by default; reduce for final calibration

# Number of spectral frames to average at each frequency step.
# More frames = lower noise, but slower sweep. 20 is a good starting point.
N_FRAMES = 20

# Normalised DAC amplitude (0.0 to 1.0). 0.5 = half full scale.
# Keep below 1.0 to avoid clipping. Keep above 0.1 for good SNR
DAC_AMPLITUDE = 0.5

# Time to wait after changing the DAC NCO  before taking a measurement (seconds)
# The DAC PLL needs time to re-lock to the new frequency. This is a starting 
# guess - Measurement 1 (switching speed test) will give the true value.
DAC_SETTLE_TIME_S = 0.1

# Output file paths (saved in the same directory as this script)
OUTPUT_NPZ = "rhino_cw_results.npz"
OUTPUT_CSV = "rhino_cw_results.csv"
OUTPUT_PLOT = "rhino_cw_gain_curve.png"

# =============================================================================
# LAYER 1 : TONE GENERATOR
# This is the only layer that needs to change when switching from rfsoc_sam
# NCO to a DDS compiler hardware block. All other code stays the same.
# =============================================================================

# Global handle for the overlay and transmitter - set up in initialise_hardware()
_overlay = None
_transmitter = None
_receiver = None

def initialise_hardware():
    """
    Load the rfsoc_sam overlay and get handles to the transmitter and receiver.
    Must be called once before any other function.
    """
    global _overlay, _transmitter, _receiver

    if not BOARD_CONNECTED:
        # Simulation mode : nothing to initialise
        print("[SIM] Hardware initialisation skipped (simulation mode).")
        return
    
    print ('[INFO] Loading rfsoc_sam overlay...')
    _overlay = Overlay("rfsoc_sam.bit")

    # channel_22 is the confirmed ADC channel for the ADC_A SMA connector 
    # This was verified in Session 1 
    _receiver = _overlay.radio.receiver.channel_22

    # The transmitter channel connected to DAC_A/ DAC_B needs to be confirmed
    # on your specific board. Replace 'channel_X' with the correct channnel.
    # TODO: update this once DAC transmitter channel is confirmed.
    _transmitter = _overlay.radio.transmitter.channel_00

    print("[INFO] Hardware initialised. Receiver: channel_22, Transmitter: channel_00")

def set_tone(freq_target_mhz, amplitude=DAC_AMPLITUDE):
    """
    Set the DAC to generate a CW tone that appears at freq_target_mhz MHz
    in the ADC output spectrum.

    """
    # Apply the 2x correction: divide target by 2 to get the correct offset
    f_offset_mhz = freq_target_mhz / 2.0
    f_dac_mhz = DDC_NCO_MHZ + f_offset_mhz

    if not BOARD_CONNECTED:
        print(f"[SIM] set_tone: target={freq_target_mhz:.3f} MHz "
              f"offset={f_offset_mhz:.3f} MHz "
              f"DAC_NCO={f_dac_mhz:.3f} MHz "
              f"amplitude={amplitude:.2f}")
        return
    
    # Set the DAC NCO frequency using the 2x- corrected value.
    # UpdateEvent(1) is required to push the new settings to the hardware -
    # without it the change is buffered but never applied to the DAC tile.
    _transmitter.dac_block.MixerSettings['Freq'] = f_dac_mhz

    # UpdateEvent(1) tells the xrfdc driver to apply the new MixerSettings.
    # Without this call, the change does not take effect.
    _transmitter.dac_block.UpdateEvent(1)

    # Set the output amplitude
    _transmitter.amplitude= amplitude

    # Enable the transmitter (it may have been disabled between steps)
    _transmitter.transmit_enable = True

    # Wait for the DAC PLL to re-lock to the new frequency.
    # The correct value for this sleep is determined by Measurement 1 (switching speed test).
    time.sleep(DAC_SETTLE_TIME_S)

def disable_tone():
    """
    Turn off the DAC transmitter. Used for noise floor measurements.
    """
    if not BOARD_CONNECTED:
        print("[SIM] disable_tone: DAC transmitter disabled.")
        return
    _transmitter.transmit_enable = False
    time.sleep(0.05) # brief settle after disabling 

# =============================================================================
# LAYER 2: SPECTRUM CAPTURE
# Reads the ADC spectrum from rfsoc_sam and averages multiple frames.
# This layer is independent of how the tone is generated.
# =============================================================================

def capture_spectrum(n_frames=N_FRAMES):
    """
    Capture and average n_frames IQ spectral frames from channel_22.

    Returns
    -------
    spectrum_dbfs : np.ndarray
        Averaged IQ power spectrum in dBFS.
    freqs_mhz : np.ndarray
         Frequency axis in MHz ( 0 to ADC_FS_HZ/2 )

    """
    if not BOARD_CONNECTED:
        # Simulation mode: return a synthetic spectrum with a tone peak
        # at a random frequenct for testing pipeline
        freqs_mhz = np.linspace (0, ADC_FS_HZ / 2e6, N_FFT // 2)
        spectrum = np.random.normal ( -107.0, 0.5, N_FFT // 2)
        #  Add a simulated tone peak ( the rfsoc_sam DDC Wwill place it at the
        #  target frequency - for simulation we just pick the midpoint)
        peak_bin = N_FFT // 4
        spectrum[peak_bin] = -20.0 # simulated tone ay -20 dBFS
        return spectrum, freqs_mhz



# =============================================================================
# LAYER 3: RESULTS LOGGER
# Saves all measurements to disk. Independent of both other layers.
# =============================================================================

 

# ============================================================================= 
# MEASUREMENT FUNCTIONS
# Each measurement is a self-contained function. They call Layer 1 and Layer 2
# but do not depend on each other.
# =============================================================================





# =============================================================================
# PLOTTING
# =============================================================================





# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
