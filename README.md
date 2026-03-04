# Voltage Imaging Processing Execution Routines (VIPER) 
Offline Pipeline for Voltage Image Processing: https://docs.google.com/presentation/d/1i1Eehe1vd-wUmb5itjOlRDMwH4RwgGJKw7-UGQe_Prk/edit?usp=sharing

To utilize the scripts in the repository download Python.

Clone the repository on your device, cd into it, and create a virtual envirnoment: python -m venv /path/to/new/virtual/environment

Activate the environment by running: source /path/to/new/virtual/environment/venv/bin/activate

Install packages with: pip install -r requirements.txt

Finally, run the code on the command line by running the GUI script: python ~/PATH/TO/FILE/GUI_[version].py

# VIPER-Mac-Port
This version has been refactored and optimized for local execution on Apple Silicon (M Chips) hardware:

Hardware Acceleration: Integrated PyTorch MPS (Metal Performance Shaders) to offload neural network segmentation (Cellpose) to the Mac GPU, bypassing legacy CUDA dependencies.

Out-of-Core Processing: Implemented a chunked memory accumulator using Zarr to process datasets exceeding 100,000+ frames without exceeding system RAM limits.

Asynchronous I/O: Utilizes dedicated QThreads for TIF ingestion and registration to ensure the PyQt5/PyQtGraph interface remains responsive during heavy computation.

Numerical Stability: Uses a float64 accumulation buffer to prevent precision overflow during large-scale temporal averaging of float16 raw data.

Normalization & Visualization: Resolved a "white rectangle" rendering bug caused by bit-depth dynamic range mismatches on macOS by implementing a 99.9th percentile normalization algorithm, making voltage spikes and neural activity visible on standard displays.

Automated Ingestion: Refactored the file import logic to automatically detect and stack disparate TIFF files into a unified 3D volume, preventing system crashes during non-stacked data loads.

Dependency Refactoring: Stripped 20+ Windows/NVIDIA-specific version locks (CUDA, CuPy, pywin32) and refactored the environment for cross-platform portability, reducing the setup time from hours to minutes.

### 1. Setup Environment 
```bash
git clone [https://github.com/daohhuynh/VIPER-Mac-Port.git](https://github.com/daohhuynh/VIPER-Mac-Port.git) 
cd VIPER-Mac-Port 
python -m venv venv 
source venv/bin/activate
```
### 2. Install & Launch
```bash
pip install -r requirements.txt
python gui.py
```

Lab Context

Developed for the Golshani Lab at UCLA David Geffen School of Medicine to enable native, offline processing of voltage imaging data. Includes specific protocols for habituation, treadmill rigging, and behavioral data collection.
