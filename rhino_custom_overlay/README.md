# RHINO Custom CASPER Overlay

**Status: in progress**

The custom CASPER overlay design (FFT 16384, 4-tap PFB, custom-designed windows, 10-100 kHz channels via RFDC decimation), built to allow a direct hardware-vs-software comparison against the QICK-based spectrometer in [`qick_overlay/`](../qick_overlay/).

The algorithmic starting point for this work is the validated reference model in [`rhino_pfb_reference_model/`](../rhino_pfb_reference_model/), currently using a placeholder window pending the group's final window design.

This folder will be populated with the Vivado block design, HDL/Model Composer sources, and generated bitstreams as that work progresses.
