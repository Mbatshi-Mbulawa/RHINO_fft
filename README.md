# RHINO 21cm Global Signal Acquisition System

**Author:** Mbatshi Jerry Junior Mbulawa  
**Supervisor:** Dr. Phil Bull  
**Institution:** University of Manchester  
**Date:** February 2025

---

## Project Overview

This repository contains the acquisition and processing pipeline for the RHINO (Radio Hydrogen Intensity Observing for the Neutral gas) 21cm global signal experiment. The system uses the Xilinx RFSoC 4x2 board to capture, process, and analyze radio frequency spectra in the 60-85 MHz band.

## Features

- **Dual Processing Modes:** Hardware FFT and software Polyphase Filter Bank (PFB)
- **Direct RF Sampling:** 4915.2 MSPS native ADC rate
- **Network Transmission:** WebSocket-based data streaming to logging server
- **DAC Tone Generation:** Test signal generation for calibration
- **Visualization:** ASCII terminal plots and matplotlib figures

## Repository Structure
```
RHINO_fft/
├── scripts/              # Python acquisition and processing scripts
├── notebooks/            # Jupyter notebooks for analysis
├── documentation/        # Technical documentation and guides
└── vivado_designs/       # FPGA block diagrams and configurations
```

## Quick Start

### Prerequisites

- RFSoC 4x2 board with PYNQ image
- Python 3.8+
- Required packages: `numpy`, `scipy`, `matplotlib`, `websockets`

### Installation
```bash
git clone https://github.com/Mbatshi-Mbulawa/RHINO_fft.git
cd RHINO_fft
pip install numpy scipy matplotlib websockets
```

### Running Acquisition
```bash
# On logging computer:
python3 scripts/rhino_data_server.py

# On RFSoC board:
python3 scripts/daq_unified_updated.py
```

## Documentation

- **[DAQ_DOCUMENTATION.md](documentation/DAQ_DOCUMENTATION.md)** - Complete code reference
- **[VIVADO_CLOCK_FIX.md](documentation/VIVADO_CLOCK_FIX.md)** - Vivado block diagram guide
- **[ACTION_ITEMS.md](documentation/ACTION_ITEMS.md)** - Current priorities

## Acknowledgments and Credits

This project builds upon and references work from the following repositories:

### Base Libraries and Tools

- **rfsoc_sam** by University of Strathclyde  
  https://github.com/strath-sdr/rfsoc_sam  
  Used for: FPGA overlay interface and spectrum analyzer functionality

- **RHINO DAQ** by RHINO Experiment Team  
  https://github.com/RHINO-Experiment/rhino-daq  
  Used for: Data acquisition pipeline architecture and network transmission

### Original Contributions

The following components were developed specifically for this thesis:
- Unified acquisition system (`scripts/Daq_unified_updated.py`)
- Polyphase Filter Bank implementation (`scripts/Daq_pfb.py`)
- DAC tone generation for calibration
- Comprehensive documentation and guides
- Vivado block diagram design

## References

1. Price, D.C. (2016). "Spectrometers and Polyphase Filterbanks in Radio Astronomy." arXiv:1607.03579
2. Bull et al. (2024). "RHINO: A large horn antenna for detecting the 21cm global signal." arXiv:2410.00076
3. University of Strathclyde. rfsoc_sam library. https://github.com/strath-sdr/rfsoc_sam
4. RHINO Experiment. rhino-daq. https://github.com/RHINO-Experiment/rhino-daq

## License

This project is part of an MSc thesis at the University of Manchester. Components derived from other repositories maintain their original licenses.

## Contact

For questions about this work:  
Mbatshi Jerry Junior Mbulawa  
Email: mbatshi.mbulawa@postgrad.manchester.ac.uk  
Supervisor: Dr. Phil Bull