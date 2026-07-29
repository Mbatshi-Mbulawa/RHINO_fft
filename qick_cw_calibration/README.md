# QICK CW Calibration Data

**Status:** raw observation data, pending write-up

Continuous-wave calibration data collected on the QICK-based software spectrometer (distinct from `rhino_cw_calibration/`, which covers the rfsoc_sam-based absolute radiometer calibration work).

## Contents
- `m5_*` — baseline/spectra/timing captures from a specific calibration run
- `rfi_*` — RFI baseline spectra and frequency axes
- `sinc_20ch_*` — 20-channel sinc-response characterisation (frequencies, offsets, power)
- `qick_dual_spectrum_*` — dual-spectrum comparison plots
- `DAC_stabilisation/` — a later DAC stabilisation run, same measurement types
- `power_meter_obs/` — power meter cross-check readings alongside the RFSoC captures

No processing scripts are included here yet, this is the raw data as pulled from the board's SD card.
