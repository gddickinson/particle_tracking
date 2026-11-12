"""Write tests/fixtures/engine_cases.json: track CSVs and what the Python analysis (web/analysis.py, i.e.
MobilityAnalyzer) returns for them, so static/engine.test.mjs can check the JS port value by value.

Cases: tracks from the repo's TrajectorySimulator (seed 7, all five motion models, rows shuffled so the per-track
order and frame sorting are exercised) plus edge tracks (too short for alpha, stationary, a straight line), analysed
with the default units/thresholds and with other units/thresholds.

Run: python scripts/make_engine_fixture.py  (tests/test_web.py checks the committed file is current)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.trajectory_simulator import TrajectorySimulator  # noqa: E402
from web.analysis import DEFAULTS, analyse, read_tracks_csv  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "engine_cases.json"


def case_csv() -> str:
    sim = TrajectorySimulator(seed=7)
    df = sim.generate_mixed_population(40, 60, dt=0.1, pixel_size=0.065)
    df["particle"] = "p" + df["particle"].astype(str)
    extra = [
        pd.DataFrame({"particle": "short", "frame": np.arange(5), "x": [1.0, 2.0, 2.5, 3.0, 3.2],
                      "y": [1.0, 1.5, 1.2, 2.0, 2.2]}),
        pd.DataFrame({"particle": "still", "frame": np.arange(20), "x": 7.0, "y": 9.0}),
        pd.DataFrame({"particle": "line", "frame": np.arange(30), "x": 3.0 + 0.5 * np.arange(30), "y": 4.0}),
    ]
    for e in extra:
        e["true_mobility"] = "edge"
    df = pd.concat([df] + extra, ignore_index=True)
    df[["x", "y"]] = df[["x", "y"]].round(9)
    rng = np.random.default_rng(3)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    return df[["particle", "frame", "x", "y", "true_mobility"]].to_csv(index=False)


def main():
    csv = case_csv()
    df = read_tracks_csv(csv.encode())
    alt = {**DEFAULTS, "coord_unit": "um", "frame_interval_s": 0.05, "subdiffusion_threshold": 0.7,
           "superdiffusion_threshold": 1.3, "confinement_threshold": 0.15, "directedness_threshold": 0.4}
    runs = [analyse(df, p, "cases.csv") for p in (dict(DEFAULTS), alt)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"csv": csv, "runs": runs}, indent=0, allow_nan=False) + "\n")
    print(f"wrote {OUT}: {df['particle'].nunique()} tracks, {len(runs)} runs")


if __name__ == "__main__":
    main()
