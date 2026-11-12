"""
Mobility Analyzer
=================

Advanced mobility analysis for fluorescently-tagged proteins with mixed behaviors.

Classifies motion into:
- Brownian diffusion (normal random walk)
- Confined diffusion (corralled/trapped)
- Directed motion (active transport)
- Subdiffusion (anomalous, α < 1)
- Superdiffusion (anomalous, α > 1)

Methods:
- Mean Square Displacement (MSD) analysis
- Anomalous exponent (α) estimation
- Diffusion coefficient calculation
- Confinement detection
- Directedness analysis
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional

from mobility_metrics import (CONFINEMENT_LEVEL_THRESHOLD, confinement_level,
                              frame_msd, frame_span)


class MobilityAnalyzer:
    """
    Analyze and classify particle mobility from trajectories
    """
    
    def __init__(self):
        """Initialize mobility analyzer with classification thresholds.

        Threshold explanations:
        - brownian_alpha_range: Anomalous exponent alpha in (0.9, 1.1) indicates
          normal Brownian diffusion (alpha=1 is pure Brownian).
        - subdiffusion_threshold: alpha < 0.9 indicates subdiffusive motion
          (e.g., crowded environments, viscoelastic media).
        - superdiffusion_threshold: alpha > 1.1 indicates superdiffusive motion
          (e.g., active transport along cytoskeleton).
        - confinement_level_threshold: Saxton/Simson confinement level
          -log10(psi) above which a track is confined (3.16, psi < ~0.07 %;
          see mobility_metrics.confinement_level and calculate_confinement_level).
        - confinement_threshold: LEGACY, no longer used for classification.
          Ratio of end-to-end distance to Rg*sqrt(track_length) < 0.3; this
          ratio falls as 1/sqrt(n) for any motion (calculate_confinement_ratio).
        - directedness_threshold: Straightness index (end-to-end / path length).
          Values > 0.5 suggest directed/processive motion.
        """
        self.brownian_alpha_range = (0.9, 1.1)  # alpha for Brownian motion
        self.subdiffusion_threshold = 0.9
        self.superdiffusion_threshold = 1.1
        self.confinement_threshold = 0.3  # LEGACY radius-of-gyration ratio (not used to classify)
        self.confinement_level_threshold = CONFINEMENT_LEVEL_THRESHOLD
        self.directedness_threshold = 0.5  # Straightness threshold
    
    def calculate_msd(self, tracks: pd.DataFrame,
                     pixel_size: float = 0.065,
                     fps: float = 10.0,
                     max_lagtime: int = 100,
                     ensemble: bool = True) -> pd.DataFrame:
        """
        Calculate Mean Square Displacement
        
        Parameters
        ----------
        tracks : pd.DataFrame
            Particle trajectories with columns ['frame', 'x', 'y', 'particle']
        pixel_size : float
            Pixel size in microns
        fps : float
            Frame rate (Hz)
        max_lagtime : int
            Maximum lag time in frames
        ensemble : bool
            If True, return ensemble-averaged MSD. If False, individual MSDs
            
        Returns
        -------
        msd : pd.DataFrame
            MSD data indexed by lag time
        """
        # Validate inputs to prevent division-by-zero
        if pixel_size <= 0:
            raise ValueError(f"pixel_size must be > 0, got {pixel_size}")
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")

        try:
            import trackpy as tp

            # Convert positions to microns
            tracks_microns = tracks.copy()
            tracks_microns['x'] = tracks['x'] * pixel_size
            tracks_microns['y'] = tracks['y'] * pixel_size
            
            if ensemble:
                # Ensemble-averaged MSD
                msd = tp.emsd(
                    tracks_microns,
                    mpp=1.0,  # Already in microns
                    fps=fps,
                    max_lagtime=max_lagtime
                )
            else:
                # Individual trajectory MSDs
                msd = tp.imsd(
                    tracks_microns,
                    mpp=1.0,
                    fps=fps,
                    max_lagtime=max_lagtime
                )
            
            return msd
            
        except ImportError:
            # Fallback MSD calculation
            return self._calculate_msd_manual(tracks, pixel_size, fps, max_lagtime)
    
    def _calculate_msd_manual(self, tracks: pd.DataFrame,
                             pixel_size: float,
                             fps: float,
                             max_lagtime: int) -> pd.DataFrame:
        """Manual MSD calculation without trackpy"""
        dt = 1.0 / fps
        
        # Prepare output
        msds = []
        
        for particle_id in tracks['particle'].unique():
            particle = tracks[tracks['particle'] == particle_id].sort_values('frame')
            
            x = particle['x'].values * pixel_size
            y = particle['y'].values * pixel_size
            
            # MSD at lags measured in frames (gaps in the frame numbers allowed)
            lags, values = frame_msd(particle['frame'].values, x, y, max_lagtime - 1)
            for lag, msd_val in zip(lags, values):
                msds.append({
                    'lagtime': lag * dt,
                    'msd': msd_val,
                    'particle': particle_id
                })
        
        msd_df = pd.DataFrame(msds)
        
        # Ensemble average
        ensemble_msd = msd_df.groupby('lagtime')['msd'].mean()
        
        return pd.DataFrame({'msd': ensemble_msd})
    
    def fit_msd_model(self, msd: pd.DataFrame,
                     model: str = 'powerlaw') -> Dict[str, float]:
        """
        Fit MSD to theoretical model
        
        Parameters
        ----------
        msd : pd.DataFrame or pd.Series
            MSD data
        model : str
            'powerlaw': MSD = 4D*t^α (general anomalous diffusion)
            'brownian': MSD = 4D*t (normal diffusion)
            'confined': MSD = L²(1 - exp(-4Dt/L²)) (confined diffusion)
            
        Returns
        -------
        params : dict
            Fitted parameters
        """
        t = msd.index.values
        
        # Handle both Series and DataFrame
        if isinstance(msd, pd.Series):
            msd_vals = msd.values
        elif 'msd' in msd.columns:
            msd_vals = msd['msd'].values
        else:
            # Take first column
            msd_vals = msd.iloc[:, 0].values if len(msd.columns) > 0 else msd.values
        
        # Remove NaN and zero values
        valid = ~np.isnan(msd_vals) & (msd_vals > 0) & (t > 0)
        t = t[valid]
        msd_vals = msd_vals[valid]
        
        if len(t) < 3:
            return {'D': np.nan, 'alpha': np.nan}
        
        if model == 'powerlaw':
            # MSD = Γ * t^α, where Γ = 4D for α=1
            # Log-log fit: log(MSD) = log(Γ) + α*log(t)
            log_t = np.log(t)
            log_msd = np.log(msd_vals)
            
            slope, intercept, r_value, p_value, std_err = linregress(log_t, log_msd)
            
            alpha = slope
            gamma = np.exp(intercept)
            D = gamma / 4.0  # Generalized diffusion coefficient
            
            return {
                'D': D,
                'alpha': alpha,
                'gamma': gamma,
                'r_squared': r_value**2
            }
        
        elif model == 'brownian':
            # MSD = 4D*t
            slope, intercept, r_value, p_value, std_err = linregress(t, msd_vals)
            D = slope / 4.0
            
            return {
                'D': D,
                'alpha': 1.0,
                'r_squared': r_value**2
            }
        
        elif model == 'confined':
            # MSD = L²(1 - exp(-4Dt/L²))
            def confined_msd(t, L, D):
                return L**2 * (1 - np.exp(-4*D*t/L**2))
            
            try:
                # Initial guess
                p0 = [1.0, 0.1]
                popt, pcov = curve_fit(confined_msd, t, msd_vals, p0=p0, maxfev=10000)
                L, D = popt
                
                return {
                    'L': L,  # Confinement length
                    'D': D,
                    'alpha': 1.0  # Effectively Brownian within confinement
                }
            except (RuntimeError, ValueError, TypeError):
                return {'L': np.nan, 'D': np.nan, 'alpha': np.nan}
        
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def calculate_diffusion_coefficients(self, tracks: pd.DataFrame,
                                        pixel_size: float = 0.065,
                                        fps: float = 10.0,
                                        max_lagtime: int = 10) -> np.ndarray:
        """
        Calculate diffusion coefficient for each trajectory
        
        Parameters
        ----------
        tracks : pd.DataFrame
            Trajectories
        pixel_size : float
            Pixel size (microns)
        fps : float
            Frame rate (Hz)
        max_lagtime : int
            Number of lag times to fit
            
        Returns
        -------
        diffusion_coeffs : np.ndarray
            Array of diffusion coefficients (μm²/s)
        """
        diffusion_coeffs = []
        
        for particle_id in tracks['particle'].unique():
            particle = tracks[tracks['particle'] == particle_id].sort_values('frame')
            
            # Calculate MSD for this particle
            x = particle['x'].values * pixel_size
            y = particle['y'].values * pixel_size
            
            # Use first few lag times (in frames) for linear fit
            lags, msds = frame_msd(particle['frame'].values, x, y, max_lagtime)
            
            if len(msds) > 2:
                # Linear fit: MSD = 4D*t
                t = lags / fps
                slope, intercept, r, p, stderr = linregress(t, msds)
                D = slope / 4.0
                diffusion_coeffs.append(D)
            else:
                diffusion_coeffs.append(np.nan)
        
        return np.array(diffusion_coeffs)
    
    def calculate_anomalous_exponent(self, tracks: pd.DataFrame,
                                    pixel_size: float = 0.065,
                                    fps: float = 10.0) -> np.ndarray:
        """
        Calculate anomalous exponent α for each trajectory
        
        α = 1: Brownian diffusion
        α < 1: Subdiffusion
        α > 1: Superdiffusion
        
        Parameters
        ----------
        tracks : pd.DataFrame
            Trajectories
        pixel_size : float
            Pixel size (microns)
        fps : float
            Frame rate (Hz)
            
        Returns
        -------
        alphas : np.ndarray
            Array of anomalous exponents
        """
        alphas = []
        
        for particle_id in tracks['particle'].unique():
            particle = tracks[tracks['particle'] == particle_id].sort_values('frame')
            
            x = particle['x'].values * pixel_size
            y = particle['y'].values * pixel_size
            
            # Calculate MSD at frame lags up to half the trajectory
            max_lag = min(frame_span(particle['frame'].values) // 2, 50)
            lags, msds = frame_msd(particle['frame'].values, x, y, max_lag - 1)
            keep = msds > 0
            lags, msds = lags[keep], msds[keep]
            
            if len(msds) > 5:
                # Log-log fit: log(MSD) = log(Γ) + α*log(t)
                t = np.array(lags) / fps
                log_t = np.log(t)
                log_msd = np.log(msds)
                
                slope, intercept, r, p, stderr = linregress(log_t, log_msd)
                alpha = slope
                alphas.append(alpha)
            else:
                alphas.append(np.nan)
        
        return np.array(alphas)
    
    def calculate_confinement_ratio(self, tracks: pd.DataFrame,
                                    pixel_size: float = 0.065) -> np.ndarray:
        """
        LEGACY confinement ratio of each trajectory (no longer used to classify)

        ratio = end-to-end distance / (Rg * sqrt(track_length)), with Rg the
        radius of gyration. It is dimensionless (pixel_size cancels).

        Returns
        -------
        ratios : np.ndarray
            One value per particle (NaN if Rg is 0)
        """
        ratios = []
        
        for particle_id in tracks['particle'].unique():
            particle = tracks[tracks['particle'] == particle_id].sort_values('frame')
            
            x = particle['x'].values * pixel_size
            y = particle['y'].values * pixel_size
            
            # Calculate radius of gyration
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            Rg_squared = np.mean((x - x_mean)**2 + (y - y_mean)**2)
            Rg = np.sqrt(Rg_squared)
            
            # Calculate asymptotic MSD plateau
            dx = x[-1] - x[0]
            dy = y[-1] - y[0]
            end_to_end = np.sqrt(dx**2 + dy**2)
            
            # Confined if Rg is small relative to trajectory length
            # and end-to-end distance is small
            track_length = len(particle)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios.append(end_to_end / (Rg * np.sqrt(track_length)))
        
        return np.array(ratios, dtype=float)
    
    def calculate_confinement_level(self, tracks: pd.DataFrame,
                                    pixel_size: float = 0.065,
                                    fps: float = 10.0) -> np.ndarray:
        """
        Saxton/Simson confinement level of each trajectory

        L = -log10(psi), psi = probability that a free random walker with the
        track's initial-slope D (MSD frame lags 1-4) stays within R, the
        track's maximum distance from its first position, for the track's
        duration (Saxton 1993, doi:10.1016/S0006-3495(93)81548-0; Simson et
        al. 1995, doi:10.1016/S0006-3495(95)79972-6). Dimensionless and, for
        free diffusion, independent of track length. NaN if R = 0.
        """
        levels = []
        for particle_id in tracks['particle'].unique():
            particle = tracks[tracks['particle'] == particle_id].sort_values('frame')
            levels.append(confinement_level(particle['frame'].values,
                                            particle['x'].values * pixel_size,
                                            particle['y'].values * pixel_size, fps))
        return np.array(levels, dtype=float)
    
    def detect_confinement(self, tracks: pd.DataFrame,
                          pixel_size: float = 0.065,
                          fps: float = 10.0) -> np.ndarray:
        """
        Detect confined motion (Saxton/Simson probability criterion)
        
        Parameters
        ----------
        tracks : pd.DataFrame
            Trajectories
        pixel_size : float
            Pixel size (microns)
        fps : float
            Frame rate (Hz)
            
        Returns
        -------
        is_confined : np.ndarray
            Boolean array indicating confined motion
            (calculate_confinement_level > confinement_level_threshold)
        """
        levels = self.calculate_confinement_level(tracks, pixel_size, fps)
        return levels > self.confinement_level_threshold
    
    def calculate_directedness(self, tracks: pd.DataFrame,
                              pixel_size: float = 0.065) -> np.ndarray:
        """
        Calculate directedness (straightness) of trajectories
        
        Directedness = (end-to-end distance) / (total path length)
        
        Parameters
        ----------
        tracks : pd.DataFrame
            Trajectories
        pixel_size : float
            Pixel size (microns)
            
        Returns
        -------
        directedness : np.ndarray
            Array of directedness values (0 to 1)
        """
        directedness_values = []
        
        for particle_id in tracks['particle'].unique():
            particle = tracks[tracks['particle'] == particle_id].sort_values('frame')
            
            x = particle['x'].values * pixel_size
            y = particle['y'].values * pixel_size
            
            # End-to-end distance
            dx = x[-1] - x[0]
            dy = y[-1] - y[0]
            end_to_end = np.sqrt(dx**2 + dy**2)
            
            # Total path length
            step_dx = np.diff(x)
            step_dy = np.diff(y)
            step_lengths = np.sqrt(step_dx**2 + step_dy**2)
            path_length = np.sum(step_lengths)
            
            # Directedness
            if path_length > 0:
                directedness = end_to_end / path_length
            else:
                directedness = 0.0
            
            directedness_values.append(directedness)
        
        return np.array(directedness_values)
    
    def classify_mobility(self, tracks: pd.DataFrame,
                         pixel_size: float = 0.065,
                         fps: float = 10.0) -> np.ndarray:
        """
        Classify each trajectory into mobility type
        
        Categories:
        - 'brownian': Normal diffusion (α ≈ 1)
        - 'confined': Confined/corralled motion
        - 'directed': Active/directed transport
        - 'subdiffusive': Anomalous subdiffusion (α < 1)
        - 'superdiffusive': Anomalous superdiffusion (α > 1)
        
        Parameters
        ----------
        tracks : pd.DataFrame
            Trajectories
        pixel_size : float
            Pixel size (microns)
        fps : float
            Frame rate (Hz)
            
        Returns
        -------
        classes : np.ndarray
            Array of mobility class labels
        """
        # Calculate metrics
        alphas = self.calculate_anomalous_exponent(tracks, pixel_size, fps)
        is_confined = self.detect_confinement(tracks, pixel_size, fps)
        directedness = self.calculate_directedness(tracks, pixel_size)
        
        # Classify
        classes = []
        for alpha, confined, directed in zip(alphas, is_confined, directedness):
            if np.isnan(alpha):
                classes.append('unknown')
            elif confined:
                classes.append('confined')
            elif directed > self.directedness_threshold:
                classes.append('directed')
            elif alpha < self.subdiffusion_threshold:
                classes.append('subdiffusive')
            elif alpha > self.superdiffusion_threshold:
                classes.append('superdiffusive')
            else:
                classes.append('brownian')
        
        return np.array(classes)
    
    def get_mobility_summary(self, tracks: pd.DataFrame,
                            pixel_size: float = 0.065,
                            fps: float = 10.0) -> pd.DataFrame:
        """
        Get complete mobility analysis summary for all tracks
        
        Parameters
        ----------
        tracks : pd.DataFrame
            Trajectories
        pixel_size : float
            Pixel size (microns)
        fps : float
            Frame rate (Hz)
            
        Returns
        -------
        summary : pd.DataFrame
            Complete mobility analysis per particle
        """
        particle_ids = tracks['particle'].unique()
        
        # Calculate all metrics
        diffusion_coeffs = self.calculate_diffusion_coefficients(tracks, pixel_size, fps)
        alphas = self.calculate_anomalous_exponent(tracks, pixel_size, fps)
        directedness = self.calculate_directedness(tracks, pixel_size)
        classes = self.classify_mobility(tracks, pixel_size, fps)
        
        # Track lengths
        # (in the same order as particle_ids: groupby would sort by particle id)
        track_lengths = tracks.groupby('particle').size().reindex(particle_ids).values
        
        # Assemble summary
        summary = pd.DataFrame({
            'particle': particle_ids,
            'track_length': track_lengths,
            'diffusion_coefficient': diffusion_coeffs,
            'anomalous_exponent': alphas,
            'directedness': directedness,
            'mobility_class': classes
        })
        
        return summary
    
    def plot_mobility_distribution(self, summary: pd.DataFrame,
                                   metric: str = 'diffusion_coefficient'):
        """
        Plot distribution of mobility metrics colored by class
        
        Parameters
        ----------
        summary : pd.DataFrame
            Mobility summary from get_mobility_summary()
        metric : str
            Which metric to plot
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = summary['mobility_class'].unique()
        colors = {'brownian': 'blue', 'confined': 'green',
                 'directed': 'red', 'subdiffusive': 'orange',
                 'superdiffusive': 'purple', 'unknown': 'gray'}
        
        for cls in classes:
            data = summary[summary['mobility_class'] == cls][metric]
            data = data[~np.isnan(data)]
            ax.hist(data, bins=30, alpha=0.6, label=cls.capitalize(),
                   color=colors.get(cls, 'gray'))
        
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
