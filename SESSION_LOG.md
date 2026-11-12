# SESSION_LOG.md

## 2026-11-11/12: web interface and static site (branch `web-interface`)
- Fixed row-order dependence in `MobilityAnalyzer` (commit "Fix row-order dependence..."): `get_mobility_summary`
  attached track lengths to the wrong particles when the table was not sorted by particle id; `detect_confinement`
  used the first/last rows instead of the first/last frames. Regression tests: `tests/test_mobility_ordering.py`.
- Refactor (no change in results): `calculate_confinement_ratio` factored out of `detect_confinement` so the web app
  can show the ratio.
- Added the web interface (`web/`, `run_web.py`, port 8130), the JS port `web/static/engine.js` with a node test
  against Python (`tests/fixtures/engine_cases.json`), static export and Pages workflow, `tests/test_web.py`,
  README section and screenshots, INTERFACE.md, CLAUDE.md.
- Open questions for George (not changed):
  1. Confinement ratio = end-to-end / (Rg·√n) falls as 1/√n for any motion, so with the 0.3 threshold every track of
     the simulator's 200-frame example run is classed "confined" (a straight evenly spaced line is "confined" from
     132 points). Pinned by `test_straight_line_confined_from_132_points`; the Validation page shows it.
  2. MSD lags are counted in rows, not frames, so tracks with gaps (trackpy memory > 0) get wrong lag times. The web
     app warns when a file has gaps.
- `mobility_analyzer.py` was already over 500 lines (578, now 593); not split in this session.

## 2026-11-12: analysis fixes requested by George ("fix any other problems")
- **Fix: MSD lags are frame differences** (`mobility_metrics.frame_msd`, used by `calculate_diffusion_coefficients`,
  `calculate_anomalous_exponent`, `_calculate_msd_manual`; engine.js mirrors it). Tracks with gaps in their frame
  numbers (trackpy memory > 0) now get correct lag times; their D and α change. Tracks without gaps give the same
  numbers as before. A repeated frame keeps its first localisation. Tests: `tests/test_mobility_fixes.py`.
