# particle_tracking

Navigation map: @INTERFACE.md. Progress: SESSION_LOG.md.

- Tests: `python -m pytest -q tests` (trackpy-dependent tests skip without trackpy); node: `node web/static/engine.test.mjs`.
- `web/static/engine.js` must stay in step with `mobility_analyzer.py`: after changing the analysis, run
  `python scripts/make_engine_fixture.py` and the node test.
- Units: positions in µm (pixel size in µm/pixel), times in s; MobilityAnalyzer takes `pixel_size` and `fps`.
