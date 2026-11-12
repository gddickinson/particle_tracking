"""Demo datasets: synthetic mixed populations from the repo's own TrajectorySimulator (tests/trajectory_simulator.py),
labelled with the motion model each track was generated from (true_mobility).

Both use the simulator's documented example settings (seed 42, 100 particles, dt = 0.1 s, pixel size 0.065 µm,
generate_mixed_population: 40 % Brownian, 25 % confined, 15 % directed, 15 % subdiffusive fBm, 5 % superdiffusive
fBm); they differ only in track length. Coordinates are in pixels, rounded to 1e-6 px for a compact download.
"""
from functools import lru_cache

from tests.trajectory_simulator import TrajectorySimulator

from .analysis import DEFAULTS, analyse, coords, standardise

SIM = {"seed": 42, "n_particles": 100, "dt": 0.1, "pixel_size": 0.065}
DEMOS = {
    "mixed50": {"title": "Mixed population, 50-frame tracks", "n_steps": 50,
                "description": "The same simulator settings with 50-frame tracks (5 s at 10 Hz)."},
    "mixed200": {"title": "Mixed population, 200-frame tracks", "n_steps": 200,
                 "description": "100 simulated tracks of 200 frames (20 s at 10 Hz): the simulator's own example run."},
}


@lru_cache(maxsize=None)
def demo_frame(demo_id: str):
    sim = TrajectorySimulator(seed=SIM["seed"])
    df = sim.generate_mixed_population(SIM["n_particles"], DEMOS[demo_id]["n_steps"],
                                       dt=SIM["dt"], pixel_size=SIM["pixel_size"])
    df[["x", "y"]] = df[["x", "y"]].round(6)
    return standardise(df)


def demo_meta(demo_id: str) -> dict:
    d = DEMOS[demo_id]
    df = demo_frame(demo_id)
    return {"id": demo_id, "title": d["title"], "description": d["description"], "n_steps": d["n_steps"],
            "n_tracks": int(df["particle"].nunique()), "n_rows": int(len(df)),
            "coord_unit": "px", "pixel_size_um": SIM["pixel_size"], "frame_interval_s": SIM["dt"],
            "generator": "tests/trajectory_simulator.py TrajectorySimulator(seed=42).generate_mixed_population("
                         f"{SIM['n_particles']}, {d['n_steps']}, dt={SIM['dt']}, pixel_size={SIM['pixel_size']})"}


def demo_params(demo_id: str) -> dict:
    m = demo_meta(demo_id)
    return {**DEFAULTS, "coord_unit": "px", "pixel_size_um": m["pixel_size_um"],
            "frame_interval_s": m["frame_interval_s"]}


@lru_cache(maxsize=None)
def demo_result(demo_id: str) -> dict:
    r = analyse(demo_frame(demo_id), demo_params(demo_id), f"{demo_id}.csv")
    r["source"] = demo_meta(demo_id)
    return r


@lru_cache(maxsize=None)
def demo_coords(demo_id: str) -> dict:
    df = demo_frame(demo_id)
    labels = df.groupby("particle", sort=False)["true_mobility"].first()
    return {"coords": coords(df), "labels": {str(k): v for k, v in labels.items()}}
