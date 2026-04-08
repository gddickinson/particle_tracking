"""
Tests for ParticleTracker parameter validation and basic functionality.

These are smoke tests that verify the tracker initializes correctly
and validates input parameters properly.
"""

import numpy as np
import pytest

import sys
from pathlib import Path

# Add parent directory to path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from particle_tracker import ParticleTracker


class TestParticleTrackerInit:
    """Test ParticleTracker initialization."""

    def test_init_creates_instance(self):
        tracker = ParticleTracker()
        assert isinstance(tracker, ParticleTracker)

    def test_trackpy_availability_flag(self):
        tracker = ParticleTracker()
        # trackpy_available should be a boolean
        assert isinstance(tracker.trackpy_available, bool)


class TestParameterValidation:
    """Test parameter validation in detect_particles."""

    def test_diameter_too_small(self):
        tracker = ParticleTracker()
        if not tracker.trackpy_available:
            pytest.skip("trackpy not installed")
        with pytest.raises(ValueError, match="diameter"):
            tracker.detect_particles(np.zeros((5, 50, 50)), diameter=1)

    def test_negative_minmass(self):
        tracker = ParticleTracker()
        if not tracker.trackpy_available:
            pytest.skip("trackpy not installed")
        with pytest.raises(ValueError, match="minmass"):
            tracker.detect_particles(np.zeros((5, 50, 50)), minmass=-1)

    def test_invalid_percentile(self):
        tracker = ParticleTracker()
        if not tracker.trackpy_available:
            pytest.skip("trackpy not installed")
        with pytest.raises(ValueError, match="percentile"):
            tracker.detect_particles(np.zeros((5, 50, 50)), percentile=200)

    def test_negative_search_range(self):
        tracker = ParticleTracker()
        if not tracker.trackpy_available:
            pytest.skip("trackpy not installed")
        import pandas as pd
        particles = pd.DataFrame({"x": [1], "y": [1], "frame": [0]})
        with pytest.raises(ValueError, match="search_range"):
            tracker.link_particles(particles, search_range=-1)

    def test_negative_memory(self):
        tracker = ParticleTracker()
        if not tracker.trackpy_available:
            pytest.skip("trackpy not installed")
        import pandas as pd
        particles = pd.DataFrame({"x": [1], "y": [1], "frame": [0]})
        with pytest.raises(ValueError, match="memory"):
            tracker.link_particles(particles, memory=-1)


class TestFilterTracks:
    """Test track filtering."""

    def test_filter_by_min_length(self):
        tracker = ParticleTracker()
        import pandas as pd
        # Create tracks: particle 0 has 20 points, particle 1 has 5
        tracks = pd.DataFrame({
            "x": list(range(20)) + list(range(5)),
            "y": list(range(20)) + list(range(5)),
            "frame": list(range(20)) + list(range(5)),
            "particle": [0] * 20 + [1] * 5,
        })
        filtered = tracker.filter_tracks(tracks, min_length=10)
        assert filtered["particle"].nunique() == 1
        assert 0 in filtered["particle"].values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
