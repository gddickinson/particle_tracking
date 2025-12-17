"""
Particle Tracker
================

High-performance particle detection and tracking using trackpy with optimizations
for single-molecule microscopy data.

Features:
- Sub-pixel localization accuracy
- Robust linking across frames with memory
- Track filtering and quality control
- Batch processing support
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class ParticleTracker:
    """
    Particle detection and tracking engine using trackpy
    """
    
    def __init__(self):
        """Initialize particle tracker"""
        try:
            import trackpy as tp
            self.tp = tp
            # Set trackpy to quiet mode
            tp.quiet()
            self.trackpy_available = True
        except ImportError:
            print("Warning: trackpy not installed. Install with: pip install trackpy")
            self.trackpy_available = False
    
    def detect_particles(self, image_stack: np.ndarray,
                        diameter: int = 11,
                        minmass: float = 100,
                        separation: Optional[float] = None,
                        percentile: float = 64) -> pd.DataFrame:
        """
        Detect particles in image stack with sub-pixel accuracy
        
        Parameters
        ----------
        image_stack : np.ndarray
            3D array (frames, height, width) or 2D single frame
        diameter : int
            Feature diameter in pixels (should be odd)
        minmass : float
            Minimum integrated brightness for particle
        separation : float, optional
            Minimum separation between particles. Default is diameter + 1
        percentile : float
            Features must have a peak brighter than this percentile
            
        Returns
        -------
        particles : pd.DataFrame
            DataFrame with columns: ['frame', 'y', 'x', 'mass', 'size', 'ecc']
        """
        if not self.trackpy_available:
            raise ImportError("trackpy is required for particle detection")
        
        # Ensure diameter is odd
        if diameter % 2 == 0:
            diameter += 1
        
        if separation is None:
            separation = diameter + 1
        
        # Handle 2D or 3D input
        if image_stack.ndim == 2:
            image_stack = image_stack[np.newaxis, ...]
        
        print(f"Detecting particles in {image_stack.shape[0]} frames...")
        
        # Batch locate particles in all frames
        particles = self.tp.batch(
            image_stack,
            diameter=diameter,
            minmass=minmass,
            separation=separation,
            percentile=percentile,
            engine='numba'  # Use numba for speed if available
        )
        
        print(f"Detected {len(particles)} total localizations")
        
        return particles
    
    def link_particles(self, particles: pd.DataFrame,
                      search_range: float = 5.0,
                      memory: int = 3,
                      adaptive_stop: Optional[float] = None,
                      adaptive_step: float = 0.95) -> pd.DataFrame:
        """
        Link particles into trajectories across frames
        
        Parameters
        ----------
        particles : pd.DataFrame
            Detected particles from detect_particles()
        search_range : float
            Maximum distance (in pixels) particles can move between frames
        memory : int
            Maximum number of frames a particle can vanish and reappear
        adaptive_stop : float, optional
            If search_range is not enough, automatically increase until
            this many particles are linked per frame
        adaptive_step : float
            When adaptive search is used, multiply search_range by this
            factor each iteration
            
        Returns
        -------
        tracks : pd.DataFrame
            DataFrame with 'particle' column added for trajectory ID
        """
        if not self.trackpy_available:
            raise ImportError("trackpy is required for particle linking")
        
        print(f"Linking particles with search range={search_range}, memory={memory}...")
        
        # Link particles into trajectories
        tracks = self.tp.link(
            particles,
            search_range=search_range,
            memory=memory,
            adaptive_stop=adaptive_stop,
            adaptive_step=adaptive_step
        )
        
        n_tracks = tracks['particle'].nunique()
        print(f"Linked into {n_tracks} trajectories")
        
        return tracks
    
    def filter_tracks(self, tracks: pd.DataFrame,
                     min_length: int = 10,
                     max_displacement: Optional[float] = None) -> pd.DataFrame:
        """
        Filter tracks by length and other quality criteria
        
        Parameters
        ----------
        tracks : pd.DataFrame
            Trajectories from link_particles()
        min_length : int
            Minimum number of frames for valid track
        max_displacement : float, optional
            Maximum total displacement (filters out very fast particles)
            
        Returns
        -------
        filtered_tracks : pd.DataFrame
            Filtered trajectories
        """
        print(f"Filtering tracks (min_length={min_length})...")
        
        # Filter by minimum length
        track_lengths = tracks.groupby('particle').size()
        valid_particles = track_lengths[track_lengths >= min_length].index
        filtered = tracks[tracks['particle'].isin(valid_particles)].copy()
        
        # Optionally filter by max displacement
        if max_displacement is not None:
            displacements = self._calculate_total_displacement(filtered)
            valid_particles = displacements[displacements <= max_displacement].index
            filtered = filtered[filtered['particle'].isin(valid_particles)].copy()
        
        n_tracks_before = tracks['particle'].nunique()
        n_tracks_after = filtered['particle'].nunique()
        print(f"Kept {n_tracks_after}/{n_tracks_before} tracks")
        
        return filtered
    
    def _calculate_total_displacement(self, tracks: pd.DataFrame) -> pd.Series:
        """Calculate total displacement for each particle"""
        def total_disp(group):
            dx = np.diff(group['x'])
            dy = np.diff(group['y'])
            return np.sum(np.sqrt(dx**2 + dy**2))
        
        return tracks.groupby('particle').apply(total_disp)
    
    def detect_and_track(self, image_stack: np.ndarray,
                        diameter: int = 11,
                        minmass: float = 100,
                        search_range: float = 5.0,
                        memory: int = 3,
                        min_track_length: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete pipeline: detect particles and link into tracks
        
        Parameters
        ----------
        image_stack : np.ndarray
            Input image stack
        diameter : int
            Particle diameter for detection
        minmass : float
            Minimum brightness threshold
        search_range : float
            Maximum displacement between frames
        memory : int
            Gap closing parameter
        min_track_length : int
            Minimum frames per track
            
        Returns
        -------
        particles : pd.DataFrame
            All detected particles
        tracks : pd.DataFrame
            Linked and filtered trajectories
        """
        # Detect
        particles = self.detect_particles(
            image_stack,
            diameter=diameter,
            minmass=minmass
        )
        
        # Link
        tracks = self.link_particles(
            particles,
            search_range=search_range,
            memory=memory
        )
        
        # Filter
        tracks = self.filter_tracks(
            tracks,
            min_length=min_track_length
        )
        
        return particles, tracks
    
    def subtract_drift(self, tracks: pd.DataFrame) -> pd.DataFrame:
        """
        Subtract global drift from trajectories
        
        Parameters
        ----------
        tracks : pd.DataFrame
            Input trajectories
            
        Returns
        -------
        drift_corrected : pd.DataFrame
            Trajectories with drift removed
        """
        if not self.trackpy_available:
            return tracks
        
        print("Subtracting drift...")
        drift = self.tp.compute_drift(tracks)
        corrected = self.tp.subtract_drift(tracks.copy(), drift)
        
        return corrected
    
    def calculate_velocities(self, tracks: pd.DataFrame,
                            fps: float = 10.0,
                            pixel_size: float = 0.065) -> pd.DataFrame:
        """
        Calculate instantaneous velocities for each trajectory
        
        Parameters
        ----------
        tracks : pd.DataFrame
            Trajectories
        fps : float
            Frame rate (Hz)
        pixel_size : float
            Pixel size (microns)
            
        Returns
        -------
        tracks_with_velocity : pd.DataFrame
            Tracks with velocity columns added
        """
        tracks = tracks.copy()
        dt = 1.0 / fps
        
        def add_velocity(group):
            # Calculate displacements
            dx = np.diff(group['x']) * pixel_size  # microns
            dy = np.diff(group['y']) * pixel_size
            
            # Calculate velocities
            vx = dx / dt  # microns/s
            vy = dy / dt
            v = np.sqrt(vx**2 + vy**2)
            
            # Pad with NaN for first frame
            group['vx'] = np.concatenate([[np.nan], vx])
            group['vy'] = np.concatenate([[np.nan], vy])
            group['v'] = np.concatenate([[np.nan], v])
            
            return group
        
        tracks = tracks.groupby('particle', group_keys=False).apply(add_velocity)
        
        return tracks
    
    def plot_trajectories(self, tracks: pd.DataFrame,
                         colorby: str = 'particle',
                         ax=None):
        """
        Plot all trajectories
        
        Parameters
        ----------
        tracks : pd.DataFrame
            Trajectories to plot
        colorby : str
            Color trajectories by this column
        ax : matplotlib axis, optional
            Axis to plot on
        """
        if not self.trackpy_available:
            print("Plotting requires trackpy")
            return
        
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        self.tp.plot_traj(tracks, colorby=colorby, ax=ax)
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        ax.set_title(f'Particle Trajectories (n={tracks["particle"].nunique()})')
        
        return ax
    
    def get_track_statistics(self, tracks: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate summary statistics for each track
        
        Parameters
        ----------
        tracks : pd.DataFrame
            Input trajectories
            
        Returns
        -------
        stats : pd.DataFrame
            Statistics per particle: length, mean position, displacement, etc.
        """
        def track_stats(group):
            return pd.Series({
                'length': len(group),
                'mean_x': group['x'].mean(),
                'mean_y': group['y'].mean(),
                'std_x': group['x'].std(),
                'std_y': group['y'].std(),
                'displacement': np.sqrt(
                    (group['x'].iloc[-1] - group['x'].iloc[0])**2 +
                    (group['y'].iloc[-1] - group['y'].iloc[0])**2
                ),
                'mean_mass': group['mass'].mean() if 'mass' in group.columns else np.nan
            })
        
        stats = tracks.groupby('particle').apply(track_stats)
        return stats
