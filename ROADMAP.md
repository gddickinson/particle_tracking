# Advanced Particle Tracking with NanoPyx — Roadmap

## Current State
A focused particle tracking application combining NanoPyx (super-resolution, drift correction) with Trackpy (detection, linking) and custom mobility analysis. Four core modules: `main.py` (PyQt6 GUI), `nanopyx_processor.py`, `particle_tracker.py`, `mobility_analyzer.py`. Also has `trajectory_simulator.py` for synthetic data, `performance_benchmarks.py`, and `example_workflow.py`. Flat file structure with no package layout. Includes synthetic test data (`synthetic_trajectories.tif`, `synthetic_tracks_ground_truth.csv`).

## Short-term Improvements
- [ ] Add a `requirements.txt` with pinned versions (nanopyx, trackpy, PyQt6, pyqtgraph, matplotlib, pandas, scipy)
- [ ] Add error handling in `nanopyx_processor.py` when NanoPyx is not installed — graceful skip with warning
- [ ] Add parameter validation in `particle_tracker.py` (diameter must be odd, search_range > 0, etc.)
- [ ] Add progress bars in `main.py` for NanoPyx processing and tracking steps
- [ ] Add docstrings to `mobility_analyzer.py` classification functions explaining thresholds
- [ ] Validate pixel_size and fps inputs before MSD calculation to prevent division-by-zero

## Feature Enhancements
- [ ] Add interactive trajectory visualization — click a track to see its MSD curve and classification
- [ ] Support multi-channel tracking (e.g., track red and green channels independently)
- [ ] Add ensemble MSD averaging across all tracks of the same mobility class
- [ ] Implement jump distance distribution (JDD) analysis as an alternative to MSD
- [ ] Add confinement radius estimation for confined tracks
- [ ] Support ND2 file format via nd2reader (already in suggested requirements)
- [ ] Add a "compare conditions" mode: load two datasets and compare diffusion coefficient distributions

## Long-term Vision
- [ ] Integrate deep learning detection (DeepSinse) as an alternative to Trackpy's classical methods
- [ ] Add real-time acquisition mode: connect to microscope and track live
- [ ] Implement Bayesian trajectory classification (aTrack method) alongside current threshold-based approach
- [ ] Add 3D tracking support for volumetric microscopy data
- [ ] Build a headless mode for batch processing on clusters
- [ ] Publish as a pip/conda package for the microscopy community

## Technical Debt
- [ ] Organize into a proper package: `particle_tracking/core/`, `particle_tracking/gui/`, `particle_tracking/analysis/`
- [ ] `performance_benchmarks.py` and `trajectory_simulator.py` should move to a `tests/` or `benchmarks/` directory
- [ ] Add unit tests using `trajectory_simulator.py` to generate known trajectories and verify classification
- [ ] Add type hints throughout all modules
- [ ] The `quick_reference.md` duplicates README content — consolidate or remove
- [ ] Extract GUI layout code from `main.py` into separate widget modules
