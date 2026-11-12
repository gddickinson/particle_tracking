# INTERFACE.md: navigation map

## Analysis (flat modules at the repo root)
| File | Contents |
|------|----------|
| `main.py` | PyQt6 desktop GUI (image viewer, processing, tracking, mobility plots, export). Imports the three modules below. |
| `nanopyx_processor.py` | `NanoPyxProcessor`: drift correction, eSRRF, channel registration, FRC/decorrelation (NanoPyx). |
| `particle_tracker.py` | `ParticleTracker`: trackpy detection, linking, filtering, drift subtraction. |
| `mobility_metrics.py` | Pure numpy helpers: `frame_msd` (MSD at frame lags, gaps allowed), `frame_span`, `initial_slope_D`, `confinement_level` (Saxton/Simson −log10 ψ), constants `CONFINEMENT_LEVEL_THRESHOLD` (3.16), `CONFINEMENT_FIT_LAGS` (4). |
| `mobility_analyzer.py` | `MobilityAnalyzer`: `calculate_msd` (trackpy emsd/imsd or `_calculate_msd_manual`), `fit_msd_model`, `calculate_diffusion_coefficients`, `calculate_anomalous_exponent`, `calculate_confinement_level`, `calculate_confinement_ratio` (legacy), `detect_confinement`, `calculate_directedness`, `classify_mobility`, `get_mobility_summary`, `plot_mobility_distribution`. Thresholds are instance attributes. numpy/pandas/scipy only. |
| `example_workflow.py` | Scripted end-to-end example. |
| `benchmarks/performance_benchmarks.py` | Performance benchmarks. `_archive/` holds older copies. |
| `tests/trajectory_simulator.py` | `TrajectorySimulator` (Brownian, confined, directed, fBm, CTRW, `generate_mixed_population`, `generate_image_stack`), `ValidationSuite`. |
| `tests/` | pytest: `test_mobility_analyzer.py`, `test_mobility_ordering.py` (row-order regressions), `test_mobility_fixes.py` (frame lags, confinement criterion), `test_particle_tracker.py` (skips without trackpy), `test_web.py`; `fixtures/engine_cases.json`. |

## Web interface (`web/` package, `__WEBMODULE__ = web.app`)
| File | Contents |
|------|----------|
| `run_web.py` | Starts uvicorn on 127.0.0.1:8130. |
| `web/app.py` | FastAPI: `/api/info`, `/api/demo/{id}/result`, `/api/demo/{id}/tracks`, `POST /api/analyze` (upload or demo + units + thresholds); `STATIC_ENDPOINTS`. |
| `web/analysis.py` | `DEFAULTS` (thresholds read from `MobilityAnalyzer`), `COLUMNS` aliases, `parse_params`, `read_tracks_csv`/`standardise`, `analyse` (get_mobility_summary + confinement level + legacy ratio + per-class ensemble MSD), `frame_warnings`, `coords`. |
| `web/demos.py` | `DEMOS` (mixed50, mixed200) from `TrajectorySimulator(seed=42).generate_mixed_population`; `demo_frame`, `demo_meta`, `demo_result`, `demo_coords`. |
| `scripts/build_static.py` | Static site export (every STATIC_ENDPOINTS GET to JSON). |
| `scripts/screenshots.py` | Playwright smoke test + `docs/img/*.png`. |
| `scripts/make_engine_fixture.py` | Writes `tests/fixtures/engine_cases.json` from `web/analysis.py`. |
| `.github/workflows/pages.yml` | Builds and publishes the static site on push to main. |

### Front end (`web/static/`, ES modules, no build step)
| File | Contents |
|------|----------|
| `index.html`, `style.css` (shared kit theme), `style_app.css`, `icon.svg` | Shell and styles. |
| `core.js` | Kit core: `api`, router, `slider`, `download`. |
| `main.js` | Registers lazy pages, mode label, background. |
| `static_api.js` | Static-site answers: exported JSON; `analyze` runs engine.js on uploads and demo tracks. |
| `engine.js` | JS port of the analysis (`analyse`, per-track metrics, `frameMsd`, `confinementLevel`, `classify`, `reclassify`, `ensembleMsd`, `fitPowerLaw`, `linregress`, `parseCsv`, `toTracks`). Tested by `engine.test.mjs`. |
| `state.js` | Current result shared across pages; `tracksOf`. |
| `fmt.js` | Class colours/names, number formatting, units text, per-track CSV. |
| `charts.js` | SVG `histogram` (stacked by class), `scatter`, `msdPlot` (log-log), `trackSvg`; canvas `drawMap`. |
| `home.js`, `analyze.js`, `tracks.js`, `populations.js`, `validate.js`, `about.js` | Pages (`about.js`: method and about). |
