# Advanced Particle Tracking with NanoPyx

A Python application for analyzing fluorescently-tagged proteins with mixed mobility behaviors in microscopy data. This system combines NanoPyx's high-performance image processing with state-of-the-art particle tracking and mobility classification.

## Features

### Image Processing (NanoPyx)
- **Drift Correction**: Automatic correction of sample drift using cross-correlation
- **eSRRF Enhancement**: Super-resolution reconstruction with enhanced SRRF
- **Channel Registration**: Align multi-color channels
- **Quality Metrics**: FRC and decorrelation analysis for resolution estimation
- **Liquid Engine**: Adaptive optimization automatically selects fastest implementation (CPU/GPU)

### Particle Tracking (Trackpy)
- **Sub-pixel Localization**: Accurate particle detection with Gaussian fitting
- **Robust Linking**: Connect particles across frames with gap closing
- **Quality Filtering**: Remove spurious detections and short tracks
- **Drift Subtraction**: Remove global drift from trajectories

### Mobility Analysis
- **MSD Analysis**: Mean square displacement calculation and fitting
- **Diffusion Classification**: Automatic classification into:
  - Brownian diffusion (normal random walk)
  - Confined diffusion (corralled/trapped particles)
  - Directed motion (active transport)
  - Subdiffusion (anomalous, α < 1)
  - Superdiffusion (anomalous, α > 1)
- **Diffusion Coefficients**: Accurate D estimation from MSD curves
- **Anomalous Exponents**: Calculate α from power-law fitting
- **Directedness Analysis**: Quantify directed vs random motion

### Visualization (PyQt6)
- **Interactive Image Viewer**: Navigate through time-lapse data
- **Particle Overlay**: Real-time visualization of detected particles
- **Track Display**: Show particle trajectories
- **MSD Plots**: Mean square displacement analysis
- **Distribution Histograms**: Visualize mobility populations
- **Results Export**: Save analysis data and figures

## Installation

### Prerequisites
- Python 3.9 or higher
- (Optional) CUDA-capable GPU for maximum performance

### Required Packages

Create a file named `requirements.txt`:

```
# Core scientific computing
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
scikit-image>=0.19.0

# Image processing and super-resolution
nanopyx>=1.0.0

# Particle tracking
trackpy>=0.6.0

# GUI and visualization
PyQt6>=6.4.0
pyqtgraph>=0.13.0
matplotlib>=3.5.0

# Optional but recommended
numba>=0.56.0  # Accelerates trackpy
pims>=0.6.0    # For reading various image formats
nd2reader>=3.3.0  # For Nikon ND2 files
```

### Installation Steps

1. **Clone or download the project**
   ```bash
   mkdir particle_tracking_project
   cd particle_tracking_project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **For NanoPyx with GPU support**
   ```bash
   # Ensure you have appropriate CUDA toolkit installed
   pip install pyopencl  # For OpenCL support
   ```

5. **Save the Python files**
   Save the following files in your project directory:
   - `main.py` - Main GUI application
   - `nanopyx_processor.py` - NanoPyx integration
   - `particle_tracker.py` - Trackpy wrapper
   - `mobility_analyzer.py` - Mobility classification

## Usage

### Quick Start

```bash
python main.py
```

### Workflow

1. **Load Image Stack**
   - Click "Load Image Stack" and select your microscopy data (.tif, .tiff, or .nd2)
   - Supported formats: Time-lapse 2D, multi-channel, multi-position

2. **Process with NanoPyx**
   - Configure processing parameters:
     - Drift Correction: Removes sample drift
     - eSRRF: Super-resolution enhancement (magnification 1-10x)
   - Click "Process with NanoPyx"
   - Processing uses Liquid Engine for optimal performance

3. **Detect & Track Particles**
   - Adjust tracking parameters:
     - **Diameter**: Expected particle size in pixels (should be odd)
     - **Min Mass**: Brightness threshold to filter noise
     - **Search Range**: Maximum displacement between frames
     - **Memory**: Frames a particle can disappear (gap closing)
     - **Min Track Length**: Filter short, unreliable tracks
   - Click "Detect & Track Particles"

4. **Analyze Mobility**
   - Set analysis parameters:
     - **Pixel Size**: Calibration (μm/pixel)
     - **Frame Rate**: Acquisition rate (Hz)
     - **Max Lag Time**: For MSD calculation
   - Click "Analyze Mobility"
   - View results in plots and summary text

5. **Export Results**
   - Click "Export Results" to save:
     - `tracks_with_mobility.csv`: All trajectories with classifications
     - `diffusion_coefficients.csv`: D and α for each particle
     - `msd_data.csv`: MSD curves

### Example Analysis Script

For programmatic use without GUI:

```python
from nanopyx_processor import NanoPyxProcessor
from particle_tracker import ParticleTracker
from mobility_analyzer import MobilityAnalyzer
from skimage import io

# Load data
image_stack = io.imread('data/timelapse.tif')

# Process with NanoPyx
processor = NanoPyxProcessor()
processed = processor.drift_correction(image_stack)
# Optional: enhanced = processor.esrrf_enhancement(processed)

# Track particles
tracker = ParticleTracker()
particles, tracks = tracker.detect_and_track(
    processed,
    diameter=11,
    minmass=100,
    search_range=5.0,
    memory=3,
    min_track_length=10
)

# Analyze mobility
analyzer = MobilityAnalyzer()
summary = analyzer.get_mobility_summary(
    tracks,
    pixel_size=0.065,  # 65 nm
    fps=10.0  # 10 Hz
)

print(summary)

# Calculate MSD
msd = analyzer.calculate_msd(tracks, pixel_size=0.065, fps=10.0)

# Export
summary.to_csv('mobility_summary.csv', index=False)
```

## Understanding the Analysis

### Mean Square Displacement (MSD)

MSD quantifies how far particles move over time:

- **MSD = 4Dt^α** (general form)
- **α = 1**: Brownian diffusion (random walk)
- **α < 1**: Subdiffusion (hindered motion)
- **α > 1**: Superdiffusion (active or ballistic transport)
- **D**: Diffusion coefficient (μm²/s)

### Mobility Classifications

| Type | Description | Characteristics |
|------|-------------|----------------|
| **Brownian** | Normal random diffusion | α ≈ 1, linear MSD |
| **Confined** | Trapped in domains | Plateau in MSD, small Rg |
| **Directed** | Active transport | High directedness, linear displacement |
| **Subdiffusive** | Hindered diffusion | α < 0.9, anomalous MSD |
| **Superdiffusive** | Enhanced spreading | α > 1.1, super-linear MSD |

### Parameters to Optimize

**For Detection:**
- **Diameter**: Should match particle size. Too small = noise; too large = missed particles
- **Min Mass**: Higher = fewer false positives, but may miss dim particles

**For Linking:**
- **Search Range**: Maximum distance particle can move. Too small = broken tracks
- **Memory**: Allows blinking or temporary occlusion. Higher = more robust but risk wrong links

**For Analysis:**
- **Pixel Size**: Critical for accurate diffusion coefficients. Check your microscope calibration
- **Frame Rate**: Faster rates capture fast dynamics but increase photobleaching
- **Max Lag Time**: Use ~1/4 of trajectory length for reliable MSD fitting

## Performance Tips

### NanoPyx Liquid Engine

The Liquid Engine automatically optimizes performance:
- First run benchmarks different implementations (CPU threaded, GPU, etc.)
- Stores benchmarks locally
- Automatically selects fastest method based on your hardware and data size
- Adapts to GPU load and hardware changes

**Manual benchmarking:**
```python
from nanopyx import benchmark

# Benchmark all implementations
benchmark.run_all_benchmarks()
```

### Large Datasets

For very large datasets:

1. **Process in batches**
   ```python
   batch_size = 100  # frames
   for i in range(0, len(image_stack), batch_size):
       batch = image_stack[i:i+batch_size]
       process_batch(batch)
   ```

2. **Use memory mapping**
   ```python
   import numpy as np
   # Instead of loading all at once
   memmap = np.memmap('large_data.dat', dtype='uint16', mode='r',
                      shape=(10000, 512, 512))
   ```

3. **Reduce memory usage**
   - Skip eSRRF if not needed
   - Process drift correction only
   - Use lower eSRRF magnification

## Troubleshooting

### "NanoPyx not installed"
```bash
pip install nanopyx
```

### "trackpy not installed"
```bash
pip install trackpy
```

### GPU not being used
- Check CUDA installation: `nvidia-smi`
- Install PyOpenCL: `pip install pyopencl`
- Verify in NanoPyx: Check which implementation is fastest

### Poor particle detection
- Adjust diameter (try odd values: 9, 11, 13)
- Lower minmass for dim particles
- Increase percentile to remove more background

### Broken trajectories
- Increase search_range
- Increase memory parameter
- Check for sample drift (apply drift correction)
- Reduce motion blur (faster acquisition)

### Unexpected mobility classification
- Check pixel calibration
- Verify frame rate
- Inspect individual trajectories visually
- Adjust classification thresholds in `MobilityAnalyzer`

## Citation

If you use this software, please cite:

**NanoPyx:**
```
Saraiva et al. (2023) "NanoPyx: super-fast bioimage analysis 
powered by adaptive machine learning" bioRxiv
```

**Trackpy:**
```
Trackpy v0.6 (http://soft-matter.github.io/trackpy/)
```

## License

This project combines multiple open-source libraries. Please respect their individual licenses:
- NanoPyx: Open source
- Trackpy: BSD-3-Clause
- PyQt6: GPL/Commercial

## Contributing

Contributions welcome! Areas for improvement:
- Deep learning-based particle detection
- 3D tracking support
- Real-time acquisition integration
- Additional mobility models (CTRW, FBM, etc.)
- Machine learning mobility classification

## References

1. **MINFLUX**: Balzarotti et al., Science 2017
2. **sptPALM**: Manley et al., Nature Methods 2008  
3. **Anomalous Diffusion**: Metzler et al., Physics Reports 2014
4. **Trackpy**: Allan et al., Soft Matter 2016
5. **AnDi Challenge**: Muñoz-Gil et al., Nature Communications 2021
6. **aTrack**: Hansen et al., eLife 2025

## Support

For issues and questions:
- NanoPyx: https://github.com/HenriquesLab/NanoPyx
- Trackpy: https://github.com/soft-matter/trackpy
