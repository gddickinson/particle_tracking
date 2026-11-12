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


# Saxton (1993), Biophys. J. 64:1766-1780, doi:10.1016/S0006-3495(93)81548-0: probability psi that a free
# random walker with diffusion coefficient D stays within radius R of its start for a time t,
#     log10(psi) = 0.2048 - 2.5117 * D * t / R**2
# (the leading term of the survival probability in a disc: 2.5117 * ln(10) = 5.783 = j_{0,1}^2).
# Simson, Sheets & Jacobson (1995), Biophys. J. 69:989-993, doi:10.1016/S0006-3495(95)79972-6, used this to detect
# confinement, with D from the initial slope of the MSD and R the maximum excursion, and called confinement when the
# level exceeded 3.16 (psi below 10**-3.16, about 0.07 %).
SAXTON_A, SAXTON_B = 0.2048, 2.5117
CONFINEMENT_LEVEL_THRESHOLD = 3.16
CONFINEMENT_FIT_LAGS = 4


def initial_slope_D(frames, x, y, fps, fit_lags=CONFINEMENT_FIT_LAGS):
    """D (units of x^2 per s) from a linear fit of MSD = 4 D t over frame lags 1..fit_lags (NaN if < 3 lags)."""
    lags, msds = frame_msd(frames, x, y, fit_lags)
    if len(msds) < 3:
        return np.nan
    t = lags / fps
    tm, mm = t.mean(), msds.mean()
    return np.sum((t - tm) * (msds - mm)) / np.sum((t - tm) ** 2) / 4.0


def confinement_level(frames, x, y, fps, fit_lags=CONFINEMENT_FIT_LAGS):
    """
    Saxton/Simson confinement level of a whole track: -log10(psi) with
    psi = Saxton's probability that a free walker with the track's initial-slope
    D stays within R (the track's maximum distance from its first position) for
    the track's duration t. Independent of track length for free diffusion;
    grows with duration for confined motion. NaN if R = 0 or D is undefined.
    """
    f = np.rint(np.asarray(frames, dtype=float))
    order = np.argsort(f, kind="stable")
    f, x, y = f[order], np.asarray(x, float)[order], np.asarray(y, float)[order]
    if f.size < 2:
        return np.nan
    D = initial_slope_D(f, x, y, fps, fit_lags)
    t = (f[-1] - f[0]) / fps
    r2 = np.max((x - x[0]) ** 2 + (y - y[0]) ** 2)
    if not np.isfinite(D) or r2 == 0:
        return np.nan
    return SAXTON_B * D * t / r2 - SAXTON_A
