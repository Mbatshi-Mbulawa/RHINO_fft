# RHINO PFB Reference Model

**MATLAB reference implementation of the custom CASPER overlay's 4-tap polyphase filter bank**
University of Manchester / Jodrell Bank Observatory
Author: Mbatshi Jerry Junior Mbulawa
Supervisors: Dr Phil Bull, Jordan Norris

---

## What This Is

A plain MATLAB implementation of the target design for the custom CASPER overlay (FFT size 16384, 4-tap PFB), used to validate the polyphase channelizer algorithm and its adjacent-channel isolation properties independently of Vivado, Vitis Model Composer, or any FPGA hardware.

The window function is Blackman applied to a sinc kernel, matching the actual RHINO PFB configuration, confirmed against:
- `rhino-daq/src/pfb_funcs.py`, `create_window()`, based on Danny Price's windowed-sinc approach: https://github.com/telegraphic/pfb_introduction, itself following Price, D.C. (2016), "Spectrometers and Polyphase Filterbanks in Radio Astronomy," arXiv:1607.03579
- `rhino-daq/obs_config.yaml`: `pfbParams.appliedWindow: blackman`, `nTaps: 4`

`blackman(L,'periodic')` is used deliberately rather than MATLAB's symmetric default, to match scipy's `get_window()`, which is periodic by default.

---

## Repository Structure
```
rhino_pfb_reference_model/
├── README.md
├── rhino_pfb_reference_model.m   # Core PFB + FFT channelizer, single-tone validation
├── rhino_pfb_isolation_test.m    # Adjacent-channel isolation test (strong + weak tone)
└── figures/
    ├── single_tone_bin100.png
    ├── single_tone_bin4000.png
    └── isolation_test.png
```

---

## Results Summary

| Test | Result |
|---|---|
| Single tone, bin 100 | Peak exactly at bin 100; -39.12 dB at ±1 bin; -117.01 dB at 5 bins away |
| Single tone, bin 4000 | Identical relative levels (expected: response is shift-invariant for an on-bin tone) |
| Isolation: weak tone 1 bin away, -60 dB down | Measured -38.36 dB, completely masked by the strong tone's own leakage |
| Isolation: weak tone 5 bins away, -60 dB down | Measured -60.01 dB, cleanly resolved |

Each result was independently verified twice: once in Python against the exact `create_window()` recipe, then again by running the MATLAB scripts here, with both producing identical figures to two decimal places.

**For reference, an earlier Hann-windowed-sinc version of this test (not the configured window, kept here only as a comparison point) gave -44.38 dB at ±1 bin and -108.23 dB at 5 bins away.** Blackman trades some immediate-neighbour rejection for better far-out suppression relative to Hann, a wider main lobe against lower sidelobes, exactly the expected direction for that pair of windows.

---

## How to Run

Requires MATLAB with Simulink and DSP System Toolbox (both included under the University's Total Access Headcount licence).

```matlab
rhino_pfb_reference_model
rhino_pfb_isolation_test
```

---

## Next Steps

- Map the validated algorithm onto AMD Toolbox or HDL Coder blocks once the digdev4 toolchain licensing is fully resolved
- Extend to model RFDC decimation ahead of the PFB, once the design's actual clocking parameters are confirmed
- Cross-check against `rhino-daq`'s own `pfb_filterbank()` output directly (same repo, same window), as a further independent confirmation beyond the two already done
