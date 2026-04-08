"""
Tests for MobilityAnalyzer using synthetic trajectory data.

Smoke tests verify that the analyzer runs without errors on known
trajectory types and produces reasonable classifications.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

# Add parent directory to path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from mobility_analyzer import MobilityAnalyzer


def _make_brownian_tracks(n_particles=5, n_frames=100, step_size=1.0, seed=42):
    """Generate synthetic Brownian motion tracks."""
    rng = np.random.default_rng(seed)
    records = []
    for pid in range(n_particles):
        x, y = 50.0, 50.0
        for frame in range(n_frames):
            records.append({"frame": frame, "x": x, "y": y, "particle": pid})
            x += rng.normal(0, step_size)
            y += rng.normal(0, step_size)
    return pd.DataFrame(records)


def _make_directed_tracks(n_particles=5, n_frames=100, velocity=2.0, seed=42):
    """Generate synthetic directed motion tracks."""
    rng = np.random.default_rng(seed)
    records = []
    for pid in range(n_particles):
        angle = rng.uniform(0, 2 * np.pi)
        vx, vy = velocity * np.cos(angle), velocity * np.sin(angle)
        x, y = 50.0, 50.0
        for frame in range(n_frames):
            records.append({"frame": frame, "x": x, "y": y, "particle": pid})
            x += vx + rng.normal(0, 0.1)
            y += vy + rng.normal(0, 0.1)
    return pd.DataFrame(records)


def _make_confined_tracks(n_particles=5, n_frames=100, radius=3.0, seed=42):
    """Generate synthetic confined motion tracks."""
    rng = np.random.default_rng(seed)
    records = []
    for pid in range(n_particles):
        cx, cy = 50.0, 50.0
        x, y = cx, cy
        for frame in range(n_frames):
            records.append({"frame": frame, "x": x, "y": y, "particle": pid})
            x += rng.normal(0, 1.0)
            y += rng.normal(0, 1.0)
            # Confine to circle
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist > radius:
                x = cx + (x - cx) * radius / dist
                y = cy + (y - cy) * radius / dist
    return pd.DataFrame(records)


class TestMobilityAnalyzerInit:
    """Test MobilityAnalyzer initialization."""

    def test_default_thresholds(self):
        analyzer = MobilityAnalyzer()
        assert analyzer.subdiffusion_threshold == 0.9
        assert analyzer.superdiffusion_threshold == 1.1
        assert analyzer.confinement_threshold == 0.3
        assert analyzer.directedness_threshold == 0.5


class TestInputValidation:
    """Test parameter validation."""

    def test_invalid_pixel_size(self):
        analyzer = MobilityAnalyzer()
        tracks = _make_brownian_tracks(n_particles=2, n_frames=20)
        with pytest.raises(ValueError, match="pixel_size"):
            analyzer.calculate_msd(tracks, pixel_size=0, fps=10.0)

    def test_invalid_fps(self):
        analyzer = MobilityAnalyzer()
        tracks = _make_brownian_tracks(n_particles=2, n_frames=20)
        with pytest.raises(ValueError, match="fps"):
            analyzer.calculate_msd(tracks, pixel_size=0.065, fps=0)


class TestMSDCalculation:
    """Test MSD calculation."""

    def test_msd_returns_data(self):
        analyzer = MobilityAnalyzer()
        tracks = _make_brownian_tracks(n_particles=3, n_frames=50)
        msd = analyzer.calculate_msd(tracks, pixel_size=0.065, fps=10.0, max_lagtime=10)
        assert msd is not None
        assert len(msd) > 0

    def test_msd_values_positive(self):
        analyzer = MobilityAnalyzer()
        tracks = _make_brownian_tracks(n_particles=3, n_frames=50)
        msd = analyzer.calculate_msd(tracks, pixel_size=0.065, fps=10.0, max_lagtime=10)
        if isinstance(msd, pd.DataFrame) and "msd" in msd.columns:
            assert (msd["msd"] >= 0).all()
        elif isinstance(msd, pd.Series):
            assert (msd >= 0).all()


class TestDiffusionCoefficients:
    """Test diffusion coefficient calculation."""

    def test_diffusion_coeffs_shape(self):
        analyzer = MobilityAnalyzer()
        tracks = _make_brownian_tracks(n_particles=5, n_frames=50)
        coeffs = analyzer.calculate_diffusion_coefficients(
            tracks, pixel_size=0.065, fps=10.0
        )
        assert len(coeffs) == 5

    def test_brownian_positive_diffusion(self):
        analyzer = MobilityAnalyzer()
        tracks = _make_brownian_tracks(n_particles=5, n_frames=100)
        coeffs = analyzer.calculate_diffusion_coefficients(
            tracks, pixel_size=0.065, fps=10.0
        )
        valid = coeffs[~np.isnan(coeffs)]
        assert len(valid) > 0
        assert (valid > 0).all()


class TestMobilityClassification:
    """Test mobility classification."""

    def test_classify_returns_array(self):
        analyzer = MobilityAnalyzer()
        tracks = _make_brownian_tracks(n_particles=3, n_frames=50)
        classes = analyzer.classify_mobility(tracks, pixel_size=0.065, fps=10.0)
        assert len(classes) == 3

    def test_classify_valid_labels(self):
        analyzer = MobilityAnalyzer()
        tracks = _make_brownian_tracks(n_particles=3, n_frames=50)
        classes = analyzer.classify_mobility(tracks, pixel_size=0.065, fps=10.0)
        valid_labels = {
            "brownian", "confined", "directed",
            "subdiffusive", "superdiffusive", "unknown",
        }
        for cls in classes:
            assert cls in valid_labels

    def test_directed_tracks_classification(self):
        """Directed tracks should tend to be classified as directed or superdiffusive."""
        analyzer = MobilityAnalyzer()
        tracks = _make_directed_tracks(n_particles=5, n_frames=100, velocity=3.0)
        classes = analyzer.classify_mobility(tracks, pixel_size=0.065, fps=10.0)
        # At least some should be directed or superdiffusive
        directed_like = sum(1 for c in classes if c in ("directed", "superdiffusive"))
        assert directed_like >= 1


class TestDirectedness:
    """Test directedness calculation."""

    def test_directed_high_straightness(self):
        analyzer = MobilityAnalyzer()
        tracks = _make_directed_tracks(n_particles=3, n_frames=50, velocity=5.0)
        directedness = analyzer.calculate_directedness(tracks, pixel_size=0.065)
        assert (directedness > 0.5).all()

    def test_confined_low_straightness(self):
        analyzer = MobilityAnalyzer()
        tracks = _make_confined_tracks(n_particles=3, n_frames=100, radius=2.0)
        directedness = analyzer.calculate_directedness(tracks, pixel_size=0.065)
        assert np.mean(directedness) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
