"""
Mobility metrics shared by MobilityAnalyzer (pure numpy functions)
==================================================================

- frame_msd: time-averaged MSD of one track at lags measured in FRAMES, so
  tracks with gaps in their frame numbers (e.g. trackpy memory > 0) get
  correct lag times. Without gaps this equals the usual row-lag MSD.
"""

import numpy as np


def frame_msd(frames, x, y, max_lag):
    """
    Time-averaged MSD of one track at frame lags 1..max_lag.

    Parameters
    ----------
    frames : array of frame numbers (rounded to integers). A repeated frame
        keeps its first localisation in frame order.
    x, y : positions (same length as frames), e.g. in microns
    max_lag : largest lag in frames

    Returns
    -------
    lags, msds : np.ndarray
        Lags (frames) that have at least one pair of localisations exactly
        that many frames apart, and the mean of dx^2 + dy^2 over those pairs.
    """
    f = np.rint(np.asarray(frames, dtype=float)).astype(int)
    order = np.argsort(f, kind="stable")
    f, x, y = f[order], np.asarray(x, float)[order], np.asarray(y, float)[order]
    if f.size == 0:
        return np.array([], int), np.array([], float)
    span = int(f[-1] - f[0])
    X = np.full(span + 1, np.nan)
    Y = np.full(span + 1, np.nan)
    idx = f - f[0]
    X[idx[::-1]] = x[::-1]  # reversed so the first localisation of a repeated frame wins
    Y[idx[::-1]] = y[::-1]
    lags, msds = [], []
    for lag in range(1, min(int(max_lag), span) + 1):
        d = (X[lag:] - X[:-lag]) ** 2 + (Y[lag:] - Y[:-lag]) ** 2
        d = d[~np.isnan(d)]
        if d.size:
            lags.append(lag)
            msds.append(np.mean(d))
    return np.array(lags, int), np.array(msds, float)


def frame_span(frames):
    """Number of frames covered by a track (last - first + 1)."""
    f = np.rint(np.asarray(frames, dtype=float))
    return int(f.max() - f.min()) + 1 if f.size else 0
