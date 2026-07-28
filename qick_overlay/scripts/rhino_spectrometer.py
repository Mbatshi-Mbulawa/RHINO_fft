import time
import argparse
import os
import yaml
import numpy as np
from scipy.signal import firwin, get_window

try: 
    from qick import QickSoc, AveragerProgram
    QICK_AVAILABLE = True
except ImportError:
    QICK_AVAILABLE = False  
    print( "[WARNING] QICK not importable - running in simulation/test mode only.")


def fft_buffs_to_powers(buffs, win_coeffs, nChannels, nTaps=None):

    spectra = []
    for buf in buffs:
        # Trim or zero-pad each buffer to exactly nChannels
        buf = buf[:nChannels]
        if len(buf) < nChannels:
            buf = np.pad(buf, (0, nChannels - len(buf)))
        buf = buf.astype(np.complex64)
        windowed = buf * win_coeffs
        spectrum = np.fft.fft(windowed)
        spectra.append(np.abs(spectrum) ** 2)
    
    spectra = np.array(spectra)
    averaged = np.mean(spectra, axis=0)
    return np.fft.fftshift(averaged)

#==========================================================================================================
# PFB functions (equivalent to pfb_funcs.py in the original repo)
#==========================================================================================================

def pfb_create_window(appliedWindow, nChannels, nTaps):

    win_coeffs = get_window(appliedWindow, nTaps *nChannels)
    sinc       = firwin(nTaps * nChannels,
                        cutoff=1.0 / nChannels,
                        window="rectangular")
    win_coeffs *= sinc
    return win_coeffs

def pfb_fir_frontend(x, win_coeffs, nTaps, nChannels):
    W          = x.shape[0] // nTaps // nChannels
    x_p        = x.reshape((W * nTaps, nChannels)).T
    h_p        = win_coeffs.reshape((nTaps, nChannels)).T
    x_weighted = x_p * h_p
    x_summed   = np.sum(x_weighted, axie=1)
    return x_summed

def pfb_filterbank(x, win_coeffs, nTaps, nChannels):
    x_fir = pfb_fir_frontend(x, win_coeffs, nTaps, nChannels)
    x_pfb = np.fft.fft(x_fir)
    return np.abs(x_pfb) ** 2

def pfb_buffs_to_powers(buffs, win_coeffs, nChannels, nTaps):
    required = nTaps * nChannels
    spectra = []
    for buf in buffs:
        buf = buf[:required]
        if len(buf) < required:
            buf = np.pad(buf.real, (0, required - len(buf))).astype(np.float32)
        buf = buf.astype(np.float32)
        spectra.append(pfb_filterbank(buf, win_coeffs, nTaps, nChannels))
    spectra  = np.array(spectra)
    averaged = np.mean(spectra, axis=0)
    return np.fft.fftshift(averaged)

#======================================================================================
# QICK DDR4 capture program
#====================================================================================== 

class DDR4CaptureProgram(AveragerProgram):

    def initialize(self):
        cfg = self.cfg
        self.declare_readout(
            ch     = cfg["adc_ch"],
            length = cfg["ro_length"],  
            freq   = 0,                 
            gen_ch = None
        )
        self.synci(200)
    
    def body(self):
        cfg = self.cfg
        self.trigger(
            adcs            = [cfg["adc_ch"]],
            ddr4            = True,        
            adc_trig_offset = 100
        )
        self.wait_all()
        self.sync_all(self.us2cycles(1.0))

def capture_ddr4(soc, prog, nt, fft_size):

    adc_ch   = prog.cfg["adc_ch"]
    avg_path = soc.get_cfg()["readouts"][adc_ch]["avgbuf_fullpath"]

    soc.ddr4_buf.set_switch(avg_path)
    soc.clear_ddr4()
    soc.ddr4_buf.arm(nt=nt)
    prog.acquire(soc, load_pulses=False, progress=False)
    raw = soc.ddr4_buf.get_mem(nt=nt)

    i_data = (raw[:,0] if raw.ndim == 2 else raw).astype(np.float32)
    return i_data

#===============================================================================================
# Main spectrometer function 
#===============================================================================================

def measure_spectra(soc,
                    runLength,
                    sampleIntegrationTime,
                    nChannels,
                    spectroMode,
                    nTaps,
                    appliedWindow,
                    adc_ch,
                    fs_mhz):


    fs_hz = fs_mhz * 1e6

    #--------Choose mode-----------------------------------------------------------------------
    if spectroMode == 'fft':
        win_coeffs  = get_window(appliedWindow, nChannels).astype(np.float32)
        buf_length  = nChannels
        spectrometer_func = fft_buffs_to_powers
        print(f"[INFO] FFT mode  |  nChannels={nChannels}  |  window={appliedWindow}")   
    else:
        win_coeffs   = pfb_create_window(appliedWindow, nChannels, nTaps)
        buf_length       = nChannels * nTaps  # samples per buffer in PFB mode
        spectrometer_func = pfb_buffs_to_powers
        print(f"[INFO] PFB mode  | nChannels={nChannels} | nTaps={nTaps} | window={appliedWindow}")

    nsamp = int(sampleIntegrationTime * fs_hz / buf_length)
    if nsamp < 1:
        nsamp = 1
    total_samples_per_row = nsamp * buf_length
    print(f"[INFO] nsamp={nsamp} buffers per integration")
    print(f"[INFO] Samples per row = {nsamp} × {buf_length} = {total_samples_per_row:,}")

    # ── DDR4 configuration ────────────────────────────────────────────────────
    NT = (total_samples_per_row // 256) + 10

    qick_cfg = {
        "adc_ch"    : adc_ch,
        "ro_length" : 1000,         # short tProcessor readout (not the DDR4 size)
        "reps"      : 1,
        "soft_avgs" : 1,
    }

    # Apply the critical API patch so soc is subscriptable
    soc.config = soc.get_cfg()
    prog = DDR4CaptureProgram(soc, qick_cfg)

    # ── Frequency axis ────────────────────────────────────────────────────────
    # Direct sampling: DC to Nyquist, then fftshift → centred at 0
    freqs = np.linspace(-fs_mhz / 2, fs_mhz / 2, nChannels)

    # ── Main observation loop ─────────────────────────────────────────────────
    waterfall_spectra = []
    times             = []

    t_end = time.time() + runLength
    row   = 0

    print(f"\n[INFO] Starting observation — {runLength:.1f} s total")
    print(f"[INFO] One row every {sampleIntegrationTime:.3f} s  "
          f"(~{runLength / sampleIntegrationTime:.0f} rows expected)\n")

    while time.time() < t_end:
        row_start = time.time()
        buffs     = []

        # Collect nsamp buffers, each of length buf_length
        for i in range(nsamp):
            raw    = capture_ddr4(soc, prog, NT, buf_length)
            # Slice exactly buf_length samples from the capture
            buf    = raw[:buf_length]
            buffs.append(buf)

        # Average nsamp spectra into one row
        spectrum = spectrometer_func(buffs, win_coeffs, nChannels, nTaps)
        ts       = time.time()

        waterfall_spectra.append(spectrum)
        times.append(ts)
        row += 1

        elapsed   = ts - row_start
        remaining = t_end - ts
        print(f"  Row {row:04d} | t={ts:.2f} | row_time={elapsed:.3f} s | "
              f"remaining={remaining:.1f} s")

    waterfall_spectra = np.array(waterfall_spectra, dtype=np.float32)
    times             = np.array(times,             dtype=np.float64)

    print(f"\n[INFO] Observation complete — {row} rows captured.")
    print(f"[INFO] Waterfall shape: {waterfall_spectra.shape}")
    return waterfall_spectra, times, freqs


# ══════════════════════════════════════════════════════════════════════════════
# Liveness check  (verify DDR4 is returning live data before the main loop)
# ══════════════════════════════════════════════════════════════════════════════

def check_ddr4_liveness(soc, prog, nt=50, n_samples=100):
    
    captures = []
    for i in range(3):
        raw = capture_ddr4(soc, prog, nt, n_samples)
        captures.append(raw[:n_samples].copy())
        time.sleep(0.05)

    same_01 = np.allclose(captures[0], captures[1])
    same_12 = np.allclose(captures[1], captures[2])

    if same_01 and same_12:
        print("[FAIL] DDR4 liveness check: all three captures identical — STALE BUFFER")
        print("       Check: soc.config = soc.get_cfg()  and  ddr4=True in trigger()")
        return False
    else:
        rms = np.sqrt(np.mean(captures[0] ** 2))
        print(f"[PASS] DDR4 liveness check: captures differ — LIVE DATA (RMS={rms:.2f} ADU)")
        return True


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="QICK RFSoC Spectrometer — FFT or PFB mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="obs_config_qick.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config, "r") as f:
        obs_config = yaml.safe_load(f)

    obs      = obs_config["observationParams"]
    qick_cfg = obs_config["qick"]

    runLength           = obs["runLength"]           # seconds
    obsCachePath        = obs["obsCachePath"]        # output directory
    storageMode         = obs.get("storageMode", "local")  # 'usb' or 'local'

    nChannels           = qick_cfg["nChannels"]      # FFT size
    sampleIntegrationTime = qick_cfg["sampleIntegrationTime"]  # seconds per row
    spectrometerMode    = qick_cfg["spectrometerMode"]  # 'fft' or 'pfb'
    appliedWindow       = qick_cfg.get("appliedWindow", "hann")
    adc_ch              = qick_cfg.get("adcChannel", 0)
    fs_mhz              = qick_cfg.get("fsMhz", 4423.68)
    bitfile             = qick_cfg.get("bitfile", None)

    nTaps = None
    if spectrometerMode == "pfb":
        nTaps = qick_cfg["pfbParams"]["nTaps"]
        appliedWindow = qick_cfg["pfbParams"].get("appliedWindow", appliedWindow)
    else:
        appliedWindow = qick_cfg["fftParams"].get("appliedWindow", appliedWindow)

    # ── Resolve output path ───────────────────────────────────────────────────
    if storageMode == "usb":
        # Auto-detect USB mount point
        usb_candidates = [
            p for p in ["/media/ubuntu", "/media/xilinx", "/mnt/usb"]
            if os.path.exists(p) and os.listdir(p)
        ]
        if usb_candidates:
            save_path = os.path.join(usb_candidates[0],
                                     os.listdir(usb_candidates[0])[0],
                                     obsCachePath)
            print(f"[INFO] Storage mode: USB → {save_path}")
        else:
            print("[WARNING] USB not found — falling back to local storage")
            save_path = obsCachePath
    else:
        save_path = obsCachePath
        print(f"[INFO] Storage mode: local → {save_path}")

    os.makedirs(save_path, exist_ok=True)

    # ── Check QICK is available ───────────────────────────────────────────────
    if not QICK_AVAILABLE:
        print("[ERROR] QICK not available. Run this script on the RFSoC board.")
        return

    # ── Load QICK overlay ─────────────────────────────────────────────────────
    print("[INFO] Loading QICK overlay...")
    if bitfile:
        soc = QickSoc(bitfile=bitfile)
    else:
        soc = QickSoc()

    # Critical patch: makes soc subscriptable so arm_ddr4 can set the switch
    soc.config = soc.get_cfg()

    fs_mhz = soc.adcs[adc_ch]["fs"] / 1e6  # read from hardware
    print(f"[INFO] ADC sample rate: {fs_mhz:.3f} MHz")
    print(f"[INFO] Nyquist:         {fs_mhz/2:.3f} MHz")

    # ── Liveness check ────────────────────────────────────────────────────────
    qick_prog_cfg = {
        "adc_ch"    : adc_ch,
        "ro_length" : 1000,
        "reps"      : 1,
        "soft_avgs" : 1,
    }
    prog = DDR4CaptureProgram(soc, qick_prog_cfg)

    print("\n[INFO] Running DDR4 liveness check...")
    live = check_ddr4_liveness(soc, prog)
    if not live:
        print("[ABORT] Stale buffer detected. Fix the DDR4 pipeline before observing.")
        return

    # ── Run spectrometer ──────────────────────────────────────────────────────
    waterfall, times, freqs = measure_spectra(
        soc                  = soc,
        runLength            = runLength,
        sampleIntegrationTime= sampleIntegrationTime,
        nChannels            = nChannels,
        spectrometerMode     = spectrometerMode,
        nTaps                = nTaps,
        appliedWindow        = appliedWindow,
        adc_ch               = adc_ch,
        fs_mhz               = fs_mhz,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    np.save(os.path.join(save_path, f"qick_waterfall_{ts_str}.npy"), waterfall)
    np.save(os.path.join(save_path, f"qick_times_{ts_str}.npy"),     times)
    np.save(os.path.join(save_path, f"qick_freqs_{ts_str}.npy"),     freqs)
    np.save(os.path.join(save_path, "new_data_bool.npy"),            True)

    print(f"\n[INFO] Data saved to: {save_path}")
    print(f"  qick_waterfall_{ts_str}.npy  — shape {waterfall.shape}")
    print(f"  qick_times_{ts_str}.npy      — {len(times)} timestamps")
    print(f"  qick_freqs_{ts_str}.npy      — {len(freqs)} channels, "
          f"{freqs[0]:.2f} to {freqs[-1]:.2f} MHz")


if __name__ == "__main__":
    main()


