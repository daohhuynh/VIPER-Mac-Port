# Voltage Imaging Processing Execution Routines (VIPER) 
Developed for the Golshani Lab at UCLA David Geffen School of Medicine to enable native, offline processing of voltage imaging data. 

# VIPER-Mac-Port
This repository contains a heavily refactored, high-throughput data processing pipeline engineered specifically for Apple Silicon (M-Chips). Originally built to handle multi-gigabyte in vivo imaging data, this architecture overhauls memory management, asynchronous I/O, and mathematical execution to scale massive, multi-dimensional neural imaging datasets without hardware bottlenecks.

# Out-of-Core Data Engineering & Memory Management
Chunked Memory Accumulation: Architected an out-of-core pipeline utilizing Zarr and Dask to stream, process, and re-chunk datasets exceeding 100,000+ frames (tens of gigabytes). This design completely bypasses system RAM limitations, ensuring zero Out-Of-Memory (OOM) crashes during the ingestion of extreme-scale raw TIF stacks.
Automated Data Ingestion Engine: Built a recursive parsing algorithm (TiffProcessingThread) with multiprocessing pools that automatically detects, sorts, and aggressively stacks fragmented .tif/.tiff files into a unified 3D volume, effectively resolving systemic lock-ups associated with disparate data loads.
Distributed Multiprocessing: Engineered parallel batch-processing (parallel_load.py) that divides volumetric mapping operations across available CPU cores for lightning-fast memory-mapped Zarr array initialization.
# Asynchronous Concurrency & GIL Bypassing
Event-Driven UI Architecture: Decoupled the computational backend from the frontend using PyQt5. Heavy workloads (multiprocessed ingestion, FFT registration, SVD extraction) are routed to isolated QThread workers.
Process Isolation: Leveraged isolated subprocess.Popen instances to completely bypass the Python Global Interpreter Lock (GIL). This guarantees the GUI's memory space remains protected and the pyqtgraph interface remains fluid and responsive under maximum compute load.
# Advanced Signal Processing & Algorithmic Math
Fourier Phase Correlation Registration: Engineered a custom sub-pixel image registration pipeline leveraging Discrete Fourier Upsampling and phase correlation to mathematically eliminate high-frequency motion artifacts from raw in vivo imaging data.
Edge Artifact Elimination: Implemented a Periodic Plus Smooth Image Decomposition algorithm prior to FFT execution to mathematically isolate and eliminate cross-boundary edge artifacts, ensuring pristine template matching.
Spike Detection & Signal Pre-Whitening: Developed an iterative match-filter trace extraction pipeline. Utilized Welch’s method for spectral density analysis to pre-whiten signal noise, isolating high-fidelity $\Delta F/F$ traces and accurately detecting micro-voltage spikes using 3rd-order Butterworth high-pass filters.
SVD Background Subtraction & Footprint Refinement: Utilized Singular Value Decomposition (SVD) for dynamic spatial background subtraction, paired with a regularized iterative solver (scipy.sparse.linalg.lsmr) to mathematically refine the spatial footprints of individual neurons.
# Hardware Acceleration & Numerical Stability
Apple Silicon / MPS Integration: Integrated PyTorch MPS (Metal Performance Shaders) to natively offload the deep neural network segmentation (Cellpose cyto3 model) directly to the Mac GPU, bypassing legacy CUDA dependencies and slashing inference times.
Numerical Stability: Engineered a float64 accumulation buffer to prevent catastrophic precision overflow and numerical corruption during the large-scale temporal averaging of float16 raw voltage arrays.
Dynamic Range Normalization: Resolved a critical macOS rendering bug (white-screen artifacts caused by bit-depth mismatches) by implementing a custom 99.9th percentile dynamic normalization algorithm (np.percentile(volume, 99.9)), scaling bit-depth to make micro-voltage spikes distinctly visible on standard displays.

### 1. Setup Environment 
```bash
# 1. Clone the repository
git clone https://github.com/daohhuynh/VIPER-Mac-Port.git
cd VIPER-Mac-Port 

# 2. Create and activate a virtual environment
python -m venv venv 
source venv/bin/activate

# 3. Install optimized dependencies
pip install -r requirements.txt
```
### 2. Launch
```bash
python gui.py
```
