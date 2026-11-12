"""
Regression tests for two analysis fixes (2026-11):

1. MSD lags are frame differences, not row counts, so tracks with gaps in the
   frame numbers (trackpy memory > 0) get correct lag times.
2. Confinement uses Saxton's probability that a free random walker stays
   within the observed radius (Saxton 1993, doi:10.1016/S0006-3495(93)81548-0),
   which does not depend on track length, instead of the length-dependent
   end-to-end / (Rg * sqrt(n)) ratio (kept as calculate_confinement_ratio,
   "legacy").
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mobility_analyzer import MobilityAnalyzer


def _df(traj, pid=0, frames=None):
    frames = np.arange(len(traj)) if frames is None else frames
    return pd.DataFrame({"particle": pid, "frame": frames, "x": traj[:, 0], "y": traj[:, 1]})


def _line(frames):
    frames = np.asarray(frames)
    return pd.DataFrame({"particle": 0, "frame": frames, "x": frames.astype(float), "y": 0.0})


# --- 1. frame-based lags ---------------------------------------------------------------------

GAPPED = [0, 1, 2, 4, 5, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24,
          25, 26, 28, 29, 30, 31, 32, 34, 35, 36]


def test_gapped_line_has_same_D_and_alpha_as_complete_line():
    a = MobilityAnalyzer()
    full, gapped = _line(range(37)), _line(GAPPED)
    d_full = a.calculate_diffusion_coefficients(full, pixel_size=1.0, fps=1.0)[0]
    d_gap = a.calculate_diffusion_coefficients(gapped, pixel_size=1.0, fps=1.0)[0]
    assert d_gap == pytest.approx(d_full, rel=1e-12)
    # MSD of a line at constant speed is exactly lag^2, so alpha is 2
    assert a.calculate_anomalous_exponent(gapped, pixel_size=1.0, fps=1.0)[0] == pytest.approx(2.0, rel=1e-9)


def test_manual_msd_uses_frame_lags():
    a = MobilityAnalyzer()
    msd = a._calculate_msd_manual(_line(GAPPED), pixel_size=1.0, fps=10.0, max_lagtime=5)
    np.testing.assert_allclose(msd.index.values, [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(msd["msd"].values, [1.0, 4.0, 9.0, 16.0])


def test_complete_tracks_unchanged_by_frame_lags():
    """Without gaps, frame lags equal row lags: same numbers as the row-based formula."""
    rng = np.random.default_rng(5)
    xy = np.cumsum(rng.normal(size=(40, 2)), axis=0)
    a = MobilityAnalyzer()
    d = a.calculate_diffusion_coefficients(_df(xy), pixel_size=1.0, fps=1.0)[0]
    msds = [np.mean(np.sum((xy[k:] - xy[:-k]) ** 2, axis=1)) for k in range(1, 11)]
    assert d == pytest.approx(np.polyfit(np.arange(1, 11), msds, 1)[0] / 4, rel=1e-9)
