"""Web interface for Particle Tracking: mobility analysis of track tables (FastAPI + plain ES-module front end).

Contents:
- app.py       FastAPI app: static front end, JSON API under /api, STATIC_ENDPOINTS for the GitHub Pages export
- analysis.py  runs MobilityAnalyzer (mobility_analyzer.py) on uploads/demos with explicit units (µm, s) and
               reshapes the result to JSON (per-track D, alpha, directedness, confinement level, legacy ratio; per-class ensemble MSD)
- demos.py     labelled demo tracks from tests/trajectory_simulator.py (TrajectorySimulator, seed 42)
- static/      the browser front end, including engine.js (JS port of the analysis; see INTERFACE.md)
"""
