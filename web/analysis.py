"""Run the repo's MobilityAnalyzer on a track table and reshape the result to JSON for the web front end.

Units are explicit: x / y are in pixels (with the pixel size in µm/pixel) or already in µm, and the frame interval
is in seconds. MobilityAnalyzer itself takes pixel_size (µm/pixel) and fps (Hz); here fps = 1 / frame_interval_s and,
for coordinates already in µm, pixel_size = 1. Diffusion coefficients come out in µm²/s, lag times in s.

The same calculation is ported to the browser in static/engine.js (checked against this module by
static/engine.test.mjs on tests/fixtures/engine_cases.json).
"""
import io
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mobility_analyzer import MobilityAnalyzer  # noqa: E402

CLASSES = ["brownian", "confined", "directed", "subdiffusive", "superdiffusive", "unknown"]

# MobilityAnalyzer's own thresholds (read from a fresh instance, never restated here) plus the units.
_A = MobilityAnalyzer()
DEFAULTS = {
    "coord_unit": "px",
    "pixel_size_um": 0.065,          # get_mobility_summary default (µm / pixel)
    "frame_interval_s": 0.1,         # 1 / fps, fps = 10 Hz is the get_mobility_summary default
    "subdiffusion_threshold": _A.subdiffusion_threshold,
    "superdiffusion_threshold": _A.superdiffusion_threshold,
    "confinement_level_threshold": _A.confinement_level_threshold,
    "directedness_threshold": _A.directedness_threshold,
}
D_FIT_LAGS = 10            # calculate_diffusion_coefficients(max_lagtime=10), as get_mobility_summary calls it
ENSEMBLE_MAX_LAG = 100     # calculate_msd(max_lagtime=100) default

# Accepted column names (first match wins); trackpy, TrackMate and ThunderSTORM-style spellings.
COLUMNS = {
    "particle": ["particle", "track_id", "track_number", "track", "TRACK_ID", "id"],
    "frame": ["frame", "FRAME", "t", "frame_number"],
    "x": ["x", "POSITION_X", "x_um", "x [px]"],
    "y": ["y", "POSITION_Y", "y_um", "y [px]"],
}
LABEL_COLUMN = "true_mobility"   # optional ground-truth label (written by the repo's TrajectorySimulator)


class AnalysisError(ValueError):
    """A problem with the user's file or parameters (shown to the user, HTTP 400)."""


def parse_params(form: dict) -> dict:
    """Form strings -> typed parameters (empty = default). Raises AnalysisError on bad values."""
    p = {}
    for k, d in DEFAULTS.items():
        v = form.get(k, "")
        if v in ("", None):
            p[k] = d
            continue
        if k == "coord_unit":
            if v not in ("px", "um"):
                raise AnalysisError("coord_unit must be 'px' or 'um'")
            p[k] = v
            continue
        try:
            p[k] = float(v)
        except ValueError:
            raise AnalysisError(f"{k} must be a number")
        if not math.isfinite(p[k]):
            raise AnalysisError(f"{k} must be a finite number")
    if p["coord_unit"] == "px" and p["pixel_size_um"] <= 0:
        raise AnalysisError("pixel_size_um must be > 0")
    if p["frame_interval_s"] <= 0:
        raise AnalysisError("frame_interval_s must be > 0")
    return p


def read_tracks_csv(data: bytes) -> pd.DataFrame:
    """Read a track CSV and rename its columns to particle, frame, x, y (+ true_mobility if present)."""
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as e:  # pandas raises several types for malformed text
        raise AnalysisError(f"Could not read the CSV file: {e}")
    return standardise(df)


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    cols, missing = {}, []
    for std, names in COLUMNS.items():
        hit = next((n for n in names if n in df.columns), None)
        if hit is None:
            missing.append(std)
        else:
            cols[hit] = std
    if missing:
        raise AnalysisError(f"Missing required columns: {', '.join(missing)} (found: {', '.join(map(str, df.columns))})")
    out = df[list(cols)].rename(columns=cols)
    if LABEL_COLUMN in df.columns:
        out[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str)
    for c in ("frame", "x", "y"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    bad = out[["frame", "x", "y"]].isna().any(axis=1)
    if bad.all():
        raise AnalysisError("No rows with numeric frame, x and y values.")
    out = out[~bad].reset_index(drop=True)
    out["particle"] = out["particle"].astype(str)
    return out


def clean(v):
    """JSON-safe number (NaN/inf -> None)."""
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _scale(p):
    pixel_size = 1.0 if p["coord_unit"] == "um" else p["pixel_size_um"]
    fps = 1.0 / p["frame_interval_s"]
    return pixel_size, fps


def analyzer_for(p) -> MobilityAnalyzer:
    a = MobilityAnalyzer()
    a.subdiffusion_threshold = p["subdiffusion_threshold"]
    a.superdiffusion_threshold = p["superdiffusion_threshold"]
    a.confinement_level_threshold = p["confinement_level_threshold"]
    a.directedness_threshold = p["directedness_threshold"]
    return a


def frame_warnings(df: pd.DataFrame) -> dict:
    """Tracks whose frames are not consecutive (handled: MSD lags are frame differences) or repeat a frame
    (only the first localisation of a repeated frame enters the MSD)."""
    gaps = dup = 0
    for _, g in df.groupby("particle", sort=False):
        d = np.diff(np.sort(g["frame"].to_numpy()))
        dup += bool((d == 0).any())
        gaps += bool((d > 1).any())
    return {"tracks_with_gaps": gaps, "tracks_with_repeated_frames": dup}


def ensemble(df: pd.DataFrame, classes: pd.Series, a: MobilityAnalyzer, pixel_size: float, fps: float) -> dict:
    """Per-class ensemble MSD (MobilityAnalyzer._calculate_msd_manual: unweighted mean of per-track MSDs over the
    tracks that reach each lag) and its power-law fit (fit_msd_model 'powerlaw')."""
    out = {}
    for cls in CLASSES:
        ids = classes.index[classes == cls]
        if not len(ids):
            continue
        try:
            msd = a._calculate_msd_manual(df[df["particle"].isin(ids)], pixel_size, fps, ENSEMBLE_MAX_LAG)
        except KeyError:  # no track in the class has two points: nothing to average
            continue
        if msd.empty:
            continue
        fit = a.fit_msd_model(msd, "powerlaw")
        out[cls] = {"n_tracks": int(len(ids)), "lag_s": [clean(t) for t in msd.index],
                    "msd_um2": [clean(m) for m in msd["msd"]],
                    "fit": {k: clean(fit.get(k)) for k in ("D", "alpha", "r_squared")}}
    return out


def analyse(df: pd.DataFrame, p: dict, name: str) -> dict:
    """MobilityAnalyzer.get_mobility_summary on the table with parameters p, plus the confinement level (the
    criterion), the legacy confinement ratio and the per-class ensemble MSD."""
    if df.empty:
        raise AnalysisError("The table has no rows.")
    pixel_size, fps = _scale(p)
    a = analyzer_for(p)
    s = a.get_mobility_summary(df, pixel_size, fps)
    level = a.calculate_confinement_level(df, pixel_size, fps)
    ratio = a.calculate_confinement_ratio(df, pixel_size)
    labels = df.groupby("particle", sort=False)[LABEL_COLUMN].first() if LABEL_COLUMN in df.columns else None
    tracks = []
    for i, row in s.iterrows():
        t = {"particle": str(row["particle"]), "track_length": int(row["track_length"]),
             "D": clean(row["diffusion_coefficient"]), "alpha": clean(row["anomalous_exponent"]),
             "directedness": clean(row["directedness"]), "confinement_level": clean(level[i]),
             "confinement_ratio_legacy": clean(ratio[i]),
             "mobility_class": str(row["mobility_class"])}
        if labels is not None:
            t["true_mobility"] = str(labels[row["particle"]])
        tracks.append(t)
    classes = pd.Series(s["mobility_class"].to_numpy(), index=s["particle"].to_numpy())
    return {"name": name, "params": p, "computed_by": "python",
            "n_rows": int(len(df)), "n_tracks": len(tracks), "has_labels": labels is not None,
            "d_fit_lags": D_FIT_LAGS, "ensemble_max_lag": ENSEMBLE_MAX_LAG,
            "warnings": frame_warnings(df), "tracks": tracks,
            "ensemble": ensemble(df, classes, a, pixel_size, fps)}


def coords(df: pd.DataFrame) -> dict:
    """{particle: [frames, xs, ys]} in the table's own units, each track sorted by frame."""
    out = {}
    for pid, g in df.groupby("particle", sort=False):
        g = g.sort_values("frame")
        out[str(pid)] = [g["frame"].tolist(), g["x"].tolist(), g["y"].tolist()]
    return out
