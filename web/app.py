"""FastAPI web app for Particle Tracking (mobility analysis): serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8130, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below.

The analysis is the repo's MobilityAnalyzer (see web/analysis.py); demo tracks come from its TrajectorySimulator
(web/demos.py). On the static site the same analysis runs in the browser (static/engine.js).
"""
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .analysis import (CLASSES, COLUMNS, D_FIT_LAGS, DEFAULTS, ENSEMBLE_MAX_LAG, LABEL_COLUMN, AnalysisError,
                       analyse, coords, parse_params, read_tracks_csv)
from .demos import DEMOS, demo_coords, demo_frame, demo_meta, demo_result

STATIC = Path(__file__).resolve().parent / "static"
TITLE = "Particle Tracking"
MAX_UPLOAD = 100 * 1024 * 1024

app = FastAPI(title=TITLE)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary: analysis defaults (MobilityAnalyzer), accepted columns, demo datasets."""
    return {"title": TITLE, "version": "1", "defaults": DEFAULTS, "classes": CLASSES, "columns": COLUMNS,
            "label_column": LABEL_COLUMN, "d_fit_lags": D_FIT_LAGS, "ensemble_max_lag": ENSEMBLE_MAX_LAG,
            "demos": [demo_meta(d) for d in DEMOS]}


@app.get("/api/demo/{demo_id}/result")
def demo(demo_id: str):
    """A demo dataset analysed by MobilityAnalyzer with its own units and the default thresholds."""
    if demo_id not in DEMOS:
        raise HTTPException(404, "unknown demo")
    return demo_result(demo_id)


@app.get("/api/demo/{demo_id}/tracks")
def demo_tracks(demo_id: str):
    """Track coordinates of a demo dataset ({particle: [frames, xs, ys]}, pixels) and the simulator's labels."""
    if demo_id not in DEMOS:
        raise HTTPException(404, "unknown demo")
    return demo_coords(demo_id)


@app.post("/api/analyze")
async def analyze(file: UploadFile | None = File(None), demo: str = Form(""),
                  coord_unit: str = Form(""), pixel_size_um: str = Form(""), frame_interval_s: str = Form(""),
                  subdiffusion_threshold: str = Form(""), superdiffusion_threshold: str = Form(""),
                  confinement_level_threshold: str = Form(""), directedness_threshold: str = Form("")):
    """Run MobilityAnalyzer on an uploaded track CSV (or a demo dataset) with the given units and thresholds."""
    form = dict(coord_unit=coord_unit, pixel_size_um=pixel_size_um, frame_interval_s=frame_interval_s,
                subdiffusion_threshold=subdiffusion_threshold, superdiffusion_threshold=superdiffusion_threshold,
                confinement_level_threshold=confinement_level_threshold, directedness_threshold=directedness_threshold)
    try:
        p = parse_params(form)
        if file is not None and file.filename:
            data = await file.read(MAX_UPLOAD + 1)
            if len(data) > MAX_UPLOAD:
                raise AnalysisError("File too large (limit 100 MB).")
            name = Path(file.filename).name
            df = read_tracks_csv(data)
            r = analyse(df, p, name)
            r["source"] = {"id": "upload", "title": name, "description": "Uploaded file"}
            r["coords"] = coords(df)
        elif demo in DEMOS:
            r = analyse(demo_frame(demo), p, f"{demo}.csv")
            r["source"] = demo_meta(demo)
            r["coords_ref"] = f"/demo/{demo}/tracks"
        else:
            raise AnalysisError("Choose a demo dataset or upload a CSV file.")
    except AnalysisError as e:
        raise HTTPException(400, str(e))
    return r


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info"] + [f"/demo/{d}/{k}" for d in DEMOS for k in ("result", "tracks")]
