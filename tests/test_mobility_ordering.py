"""
Regression tests: MobilityAnalyzer results must not depend on the row order of
the tracks table (trackpy output is usually sorted by frame, not by particle,
and user CSVs can be in any order).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from mobility_analyzer import MobilityAnalyzer


def _line_track(pid, n, step=1.0):
    """A straight track along x (directed, never confined)."""
    return pd.DataFrame({"particle": pid, "frame": np.arange(n),
                         "x": 10.0 + step * np.arange(n), "y": np.full(n, 5.0)})


def test_summary_track_length_matches_particle_when_ids_unsorted():
    # Particles appear in the order 2, 0, 1 with different lengths.
    tracks = pd.concat([_line_track(2, 30), _line_track(0, 12), _line_track(1, 20)],
                       ignore_index=True)
    summary = MobilityAnalyzer().get_mobility_summary(tracks, pixel_size=0.1, fps=10.0)
    counts = tracks.groupby("particle").size()
    for _, row in summary.iterrows():
        assert row["track_length"] == counts[row["particle"]]


def test_confinement_uses_first_and_last_frame_not_first_and_last_row():
    track = _line_track(0, 20)
    # Same localisations, rows reordered so that the first and last ROWS are
    # neighbouring frames (10 and 11).
    order = [10] + [i for i in range(20) if i not in (10, 11)] + [11]
    shuffled = track.iloc[order].reset_index(drop=True)
    analyzer = MobilityAnalyzer()
    expected = analyzer.detect_confinement(track, pixel_size=0.1)
    got = analyzer.detect_confinement(shuffled, pixel_size=0.1)
    assert not expected[0]
    assert list(got) == list(expected)
