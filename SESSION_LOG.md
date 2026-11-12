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
- **Fix: confinement criterion** (changes classifications by design). `detect_confinement` now uses the
  Saxton/Simson confinement level −log10 ψ = 2.5117·D·t/R² − 0.2048 > 3.16 (`mobility_metrics.confinement_level`;
  D from MSD frame lags 1–4, t = duration, R = max distance from start; Saxton 1993 doi:10.1016/S0006-3495(93)81548-0,
  Simson et al. 1995 doi:10.1016/S0006-3495(95)79972-6, both DOIs checked on Crossref; coefficients checked: 2.5117·ln10
  = 5.783 = j01², the disc survival exponent, so the log is base 10). The old end-to-end/(Rg·√n) ratio is kept as
  `calculate_confinement_ratio` ("legacy"). `confinement_threshold` (0.3) remains as an unused legacy attribute;
  new attribute `confinement_level_threshold` = 3.16. Demo, seed 42: 200-frame tracks now 68 % correct (was 25 %, all
  "confined"), confined class 25/25 with no false positives; 50-frame 61 % (was 49 %). Remaining errors are Brownian vs
  sub/superdiffusive from the α thresholds. Simplification vs Simson: whole track, not sliding windows.
- Web app: parameter `confinement_level_threshold`, columns `confinement_level` and `confinement_ratio_legacy`;
  Populations/Validation/Method pages updated. All KIT checks re-run (pytest, node, screenshots static and server).
- The 2026-11-11 open questions 1 and 2 are resolved by these fixes.
