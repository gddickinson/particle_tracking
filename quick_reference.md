# Quick Reference Guide: Typical Parameters

## Imaging Parameters by Application

### Single-Molecule Localization (PALM/STORM)
```python
# Typical settings
pixel_size = 0.100  # μm (100 nm) - common for sCMOS
frame_rate = 50.0   # Hz - fast for single molecules
diameter = 5        # pixels - small, diffraction-limited spots
minmass = 50        # Low threshold for single molecules
search_range = 2.0  # pixels - small, molecules don't move far
memory = 1          # frames - minimal blinking
```

**Expected diffusion coefficients:**
- Membrane proteins: 0.01 - 0.5 μm²/s
- Cytoplasmic proteins: 0.5 - 10 μm²/s  
- Fast-diffusing lipids: 1 - 5 μm²/s

### Live-Cell TIRF Microscopy
```python
# Typical settings
pixel_size = 0.065  # μm (65 nm) - 100x high NA objective
frame_rate = 10.0   # Hz - balance between speed and photons
diameter = 11       # pixels - larger for dimmer particles
minmass = 100       # Medium threshold
search_range = 5.0  # pixels - cells have more movement
memory = 3          # frames - allow for blinking/occlusion
```

**Expected behaviors:**
- Receptor diffusion: Brownian or confined
- Vesicle transport: Directed motion (α ≈ 1-1.5)
- Membrane domains: Confined (α < 1)

### Wide-Field Epifluorescence
```python
# Typical settings  
pixel_size = 0.108  # μm (108 nm) - typical 60x/1.4 NA
frame_rate = 5.0    # Hz - slower for dimmer fluorophores
diameter = 13       # pixels - account for out-of-focus blur
minmass = 200       # Higher to reject background
search_range = 8.0  # pixels - more drift and movement
memory = 5          # frames - more photobleaching gaps
```

**Expected behaviors:**
- Organelles: Directed or confined
- Nuclear proteins: Subdiffusive (crowded environment)
- Cytoskeletal-bound proteins: Directed transport

## Mobility Classification: Expected Values

### Brownian Diffusion
```
Anomalous exponent (α): 0.9 - 1.1
Directedness: < 0.3
MSD: Linear with time
Examples: Small molecules in solution, lipids in fluid membranes
```

### Confined Diffusion  
```
Anomalous exponent (α): 0.5 - 0.9 (but confined detection takes priority)
Directedness: < 0.3
MSD: Plateaus at confinement size
Examples: Receptors in membrane domains, proteins in nuclear compartments
```

### Directed Motion
```
Anomalous exponent (α): 1.0 - 2.0
Directedness: > 0.5
MSD: Quadratic or super-linear with time
Examples: Motor protein transport, vesicle trafficking
```

### Subdiffusion
```
Anomalous exponent (α): 0.3 - 0.9
Directedness: < 0.3
MSD: Sub-linear power law
Examples: Proteins in crowded cytoplasm, chromatin dynamics
```

### Superdiffusion
```
Anomalous exponent (α): 1.1 - 2.0  
Directedness: Variable
MSD: Super-linear power law
Examples: Lévy walks, ballistic transport
```

## Common Experimental Scenarios

### Scenario 1: Membrane Protein Tracking
```python
# Hardware
pixel_size = 0.065      # μm (100x/1.45 NA objective)
frame_rate = 20.0       # Hz (50 ms exposure)

# Detection
diameter = 9            # Small spots
minmass = 80            # Moderate threshold
separation = 11         # diameter + 2

# Linking
search_range = 4.0      # ~260 nm between frames
memory = 2              # Allow brief disappearances
min_track_length = 20   # 1 second minimum

# Analysis
max_lagtime = 50        # 2.5 seconds
```

**Expected Results:**
- D: 0.05 - 0.5 μm²/s
- 30-50% confined (lipid rafts, corrals)
- 50-70% Brownian diffusion
- <5% directed (near cell edge artifacts)

### Scenario 2: Vesicle Trafficking
```python
# Hardware
pixel_size = 0.100      # μm
frame_rate = 5.0        # Hz (200 ms exposure)

# Detection  
diameter = 15           # Larger vesicles
minmass = 300           # High SNR required
separation = 20         # Well-separated

# Linking
search_range = 10.0     # Fast movement: ~1 μm between frames  
memory = 5              # Stops and starts
min_track_length = 10   # 2 seconds

# Analysis
max_lagtime = 25        # 5 seconds
```

**Expected Results:**
- D: 0.5 - 2.0 μm²/s (when diffusing)
- 20-40% directed motion (active transport)
- 30-50% Brownian (diffusing between tracks)
- 20-30% confined (pausing at destinations)

### Scenario 3: Fast Cytoplasmic Proteins
```python
# Hardware
pixel_size = 0.080      # μm
frame_rate = 50.0       # Hz (20 ms exposure)

# Detection
diameter = 7            # Very small, fast particles
minmass = 50            # Lower threshold
separation = 9          

# Linking  
search_range = 3.0      # Fast but small movements
memory = 1              # No blinking expected
min_track_length = 50   # 1 second, many frames

# Analysis
max_lagtime = 100       # 2 seconds
```

**Expected Results:**
- D: 5 - 20 μm²/s (fast diffusion)
- 70-90% Brownian
- 5-15% subdiffusive (interactions)
- <5% directed

## Parameter Optimization Workflow

### Step 1: Optimize Detection
```python
# Test on single frame first
test_frame = image_stack[100]  # Middle frame

# Try range of diameters
for d in [7, 9, 11, 13, 15]:
    particles = tracker.detect_particles(test_frame, diameter=d, minmass=100)
    print(f"Diameter {d}: {len(particles)} particles")
    # Visually inspect results
```

**Goal:** Detect real particles, minimize false positives
- Too many particles? Increase minmass or diameter
- Missing particles? Decrease minmass or try different diameter

### Step 2: Optimize Linking
```python
# Try on small stack
small_stack = image_stack[0:50]  

# Test search ranges
for sr in [3.0, 5.0, 7.0, 10.0]:
    particles = tracker.detect_particles(small_stack, diameter=11, minmass=100)
    tracks = tracker.link_particles(particles, search_range=sr, memory=3)
    n_tracks = tracks['particle'].nunique()
    mean_length = tracks.groupby('particle').size().mean()
    print(f"Search range {sr}: {n_tracks} tracks, mean length {mean_length:.1f}")
```

**Goal:** Maximize track length, minimize fragmentation
- Many short tracks? Increase search_range or memory
- Unlikely connections? Decrease search_range

### Step 3: Validate Results
```python
# Check detection quality
particles['mass'].hist(bins=50)
# Should show clear peak above background

# Check track length distribution  
track_lengths = tracks.groupby('particle').size()
track_lengths.hist(bins=50)
# Should decay exponentially (geometric distribution)

# Visual inspection
tracker.plot_trajectories(tracks[tracks['particle'].isin(range(10))])
# Tracks should look smooth and continuous
```

## Diffusion Coefficient Reference

### Biological Molecules (Typical Values at 37°C)

| Molecule | Environment | D (μm²/s) | Type |
|----------|-------------|-----------|------|
| Small molecule (MW ~500) | Water | 500 | Brownian |
| GFP (27 kDa) | Water | 87 | Brownian |
| GFP | Cytoplasm | 20-30 | Subdiffusive |
| Typical protein | Cytoplasm | 5-15 | Subdiffusive |
| Membrane protein | Fluid membrane | 0.1-1.0 | Brownian/Confined |
| Lipid | Fluid membrane | 1-5 | Brownian |
| Receptor | Cell membrane | 0.01-0.1 | Confined |
| Vesicle (100 nm) | Cytoplasm | 0.5-2 | Brownian/Directed |
| mRNA | Nucleus | 0.05-0.5 | Subdiffusive |
| Chromatin locus | Nucleus | 0.001-0.01 | Subdiffusive |

### Physical Reference (Water at 25°C)

Calculate theoretical diffusion coefficient using Stokes-Einstein:
```
D = kT / (6πηr)

where:
k = 1.38 × 10⁻²³ J/K (Boltzmann constant)
T = 298 K (25°C)  
η = 0.89 mPa·s (water viscosity)
r = particle radius (m)
```

**Examples:**
- r = 1 nm → D ≈ 200 μm²/s
- r = 5 nm → D ≈ 40 μm²/s  
- r = 10 nm → D ≈ 20 μm²/s
- r = 50 nm → D ≈ 4 μm²/s
- r = 100 nm → D ≈ 2 μm²/s

## Troubleshooting by Symptom

### Too Many False Detections
```python
# Solutions:
minmass *= 1.5           # Increase threshold
diameter += 2            # Larger diameter filters noise
percentile = 70          # Higher percentile
```

### Missing Real Particles
```python
# Solutions:
minmass *= 0.7           # Lower threshold
diameter -= 2            # Smaller diameter
percentile = 60          # Lower percentile
separation = diameter    # Allow closer spacing
```

### Fragmented Tracks
```python
# Solutions:
search_range *= 1.5      # Allow larger movements
memory += 2              # Bridge longer gaps
min_track_length -= 5    # Keep shorter tracks initially
```

### Connected Wrong Particles
```python
# Solutions:
search_range *= 0.7      # Restrict linking distance
memory -= 1              # Less gap closing
separation *= 1.2        # Better initial separation
```

### Negative Diffusion Coefficients
```python
# Causes and solutions:
# 1. Insufficient localization precision
#    → Increase signal (longer exposure, more photons)
# 2. Localization bias (pixelation artifacts)
#    → Use sub-pixel localization (trackpy does this automatically)
# 3. Wrong pixel calibration
#    → Verify pixel_size matches microscope objective
# 4. Too short lag time
#    → Use lag times where MSD > localization precision²
```

### All Particles Classified as One Type
```python
# Solutions:
# 1. Adjust classification thresholds
analyzer.subdiffusion_threshold = 0.85  # More sensitive
analyzer.directedness_threshold = 0.6    # More stringent

# 2. Check trajectory length
#    → Need >20 points for reliable α estimation
min_track_length = 25

# 3. Verify physical parameters
#    → pixel_size and fps must be correct
```

## Best Practices Summary

✅ **Do:**
- Start with literature values for similar experiments
- Test parameters on subset before full analysis
- Visually inspect results at each step
- Save intermediate results
- Document parameter choices
- Use drift correction for long acquisitions
- Calibrate pixel size accurately
- Consider photobleaching (shorter = better for tracking)

❌ **Don't:**
- Use even-numbered diameters (trackpy requires odd)
- Set search_range < expected displacement
- Trust very short tracks (< 10 frames)
- Mix particles of very different sizes in one detection
- Forget to convert pixels to physical units
- Over-interpret single trajectories
- Ignore failed tracks in statistics

## Example: Complete Analysis
```python
# 1. Load and preprocess
image_stack = io.imread('data.tif')
processor = NanoPyxProcessor()
corrected = processor.drift_correction(image_stack)

# 2. Detect and track
tracker = ParticleTracker()
particles, tracks = tracker.detect_and_track(
    corrected,
    diameter=11,         # Optimized from test
    minmass=120,         # Optimized from test
    search_range=5.0,    # Based on expected speed
    memory=3,            # Allow 3-frame gaps
    min_track_length=15  # ~1.5 seconds at 10 Hz
)

# 3. Analyze mobility
analyzer = MobilityAnalyzer()
summary = analyzer.get_mobility_summary(
    tracks,
    pixel_size=0.065,  # 100x/1.45 NA objective
    fps=10.0           # 100 ms exposure
)

# 4. Results
print(summary['mobility_class'].value_counts())
print(f"Median D: {summary['diffusion_coefficient'].median():.3f} μm²/s")
print(f"Mean α: {summary['anomalous_exponent'].mean():.2f}")

# 5. Export
summary.to_csv('mobility_analysis.csv', index=False)
```

This should yield interpretable results with the parameters properly optimized for your specific experimental conditions!
