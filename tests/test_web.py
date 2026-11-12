"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific API tests -------------------------------------------------------------------------------------
import io  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mobility_analyzer import MobilityAnalyzer  # noqa: E402


def _csv(df):
    return {"file": ("t.csv", df.to_csv(index=False).encode(), "text/csv")}


def test_info_defaults_come_from_mobility_analyzer():
    d = client.get("/api/info").json()["defaults"]
    a = MobilityAnalyzer()
    assert d["confinement_level_threshold"] == a.confinement_level_threshold == 3.16
    assert d["directedness_threshold"] == a.directedness_threshold
    assert d["subdiffusion_threshold"] == a.subdiffusion_threshold
    assert d["superdiffusion_threshold"] == a.superdiffusion_threshold


def test_demo_result_matches_get_mobility_summary():
    from web.demos import demo_frame
    r = client.get("/api/demo/mixed50/result").json()
    s = MobilityAnalyzer().get_mobility_summary(demo_frame("mixed50"), 0.065, 10.0)
    assert r["has_labels"] and r["n_tracks"] == 100
    assert [t["mobility_class"] for t in r["tracks"]] == list(s["mobility_class"])
    np.testing.assert_allclose([t["D"] for t in r["tracks"]], s["diffusion_coefficient"], rtol=1e-12)
    tr = client.get("/api/demo/mixed50/tracks").json()
    assert set(tr["coords"]) == {t["particle"] for t in r["tracks"]}
    assert set(tr["labels"].values()) == {"brownian", "confined", "directed", "subdiffusive", "superdiffusive"}


def test_upload_units_um_and_px_agree():
    rng = np.random.default_rng(0)
    xy = np.cumsum(rng.normal(0, 2.0, size=(60, 2)), axis=0)
    px = pd.DataFrame({"TRACK_ID": 1, "FRAME": np.arange(60), "POSITION_X": xy[:, 0], "POSITION_Y": xy[:, 1]})
    um = px.assign(POSITION_X=px.POSITION_X * 0.1, POSITION_Y=px.POSITION_Y * 0.1)
    a = client.post("/api/analyze", files=_csv(px), data={"pixel_size_um": "0.1", "frame_interval_s": "0.05"}).json()
    b = client.post("/api/analyze", files=_csv(um), data={"coord_unit": "um", "frame_interval_s": "0.05"}).json()
    assert a["tracks"][0]["D"] == pytest.approx(b["tracks"][0]["D"], rel=1e-9)
    assert a["params"]["coord_unit"] == "px" and b["params"]["coord_unit"] == "um"
    assert "coords" in a and a["source"]["id"] == "upload"


def test_upload_errors_and_gap_warning():
    bad = pd.DataFrame({"particle": [1, 1], "frame": [0, 1], "x": [0.0, 1.0]})
    r = client.post("/api/analyze", files=_csv(bad))
    assert r.status_code == 400 and "Missing required columns: y" in r.text
    ok = pd.DataFrame({"particle": 1, "frame": [0, 1, 3, 4, 5], "x": [0.0, 1, 2, 3, 4], "y": 0.0})
    assert client.post("/api/analyze", files=_csv(ok), data={"frame_interval_s": "0"}).status_code == 400
    w = client.post("/api/analyze", files=_csv(ok)).json()["warnings"]
    assert w == {"tracks_with_gaps": 1, "tracks_with_repeated_frames": 0}
    assert client.post("/api/analyze", data={"demo": "nope"}).status_code == 400


def test_demo_with_other_thresholds():
    r = client.post("/api/analyze", data={"demo": "mixed200", "confinement_level_threshold": "50"}).json()
    assert r["params"]["confinement_level_threshold"] == 50 and r["coords_ref"] == "/demo/mixed200/tracks"
    assert len({t["mobility_class"] for t in r["tracks"]}) > 1


def test_demo200_confinement_matches_simulation():
    """With the Saxton/Simson criterion, the 200-frame demo's confined class is the simulated confined tracks."""
    r = client.get("/api/demo/mixed200/result").json()
    pred = {t["particle"] for t in r["tracks"] if t["mobility_class"] == "confined"}
    true = {t["particle"] for t in r["tracks"] if t["true_mobility"] == "confined"}
    assert len(pred ^ true) <= 2 and len(true) == 25
    assert all(t["confinement_ratio_legacy"] is not None for t in r["tracks"])


def test_engine_fixture_is_current(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_engine_fixture as m
    monkeypatch.setattr(m, "OUT", tmp_path / "cases.json")
    m.main()
    new = json.loads((tmp_path / "cases.json").read_text())
    old = json.loads((ROOT / "tests" / "fixtures" / "engine_cases.json").read_text())
    assert pd.read_csv(io.StringIO(new["csv"])).shape == pd.read_csv(io.StringIO(old["csv"])).shape
    for rn, ro in zip(new["runs"], old["runs"]):
        assert [t["mobility_class"] for t in rn["tracks"]] == [t["mobility_class"] for t in ro["tracks"]]
        for k in ("D", "alpha", "directedness", "confinement_level", "confinement_ratio_legacy"):
            # the stationary edge track's ratio is 0/0 up to rounding (0 or NaN depending on libm): skip it
            keep = [t["particle"] != "still" or k != "confinement_ratio_legacy" for t in ro["tracks"]]
            a = np.array([np.nan if t[k] is None else t[k] for t, ok in zip(rn["tracks"], keep) if ok], float)
            b = np.array([np.nan if t[k] is None else t[k] for t, ok in zip(ro["tracks"], keep) if ok], float)
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=k)
