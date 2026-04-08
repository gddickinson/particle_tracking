"""
Trajectory Simulator
====================

Generate synthetic particle trajectories for testing and validation.
Supports multiple mobility models:
- Brownian diffusion
- Confined diffusion
- Directed motion
- Anomalous diffusion (subdiffusion and superdiffusion)
- Fractional Brownian motion
- Continuous-time random walk

Also generates synthetic microscopy images with realistic noise.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import pandas as pd


class TrajectorySimulator:
    """
    Generate synthetic particle trajectories with various mobility patterns
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize simulator

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

    def brownian_motion(self, n_steps: int, D: float = 1.0,
                       dt: float = 0.1, dim: int = 2) -> np.ndarray:
        """
        Generate Brownian motion trajectory

        Parameters
        ----------
        n_steps : int
            Number of time steps
        D : float
            Diffusion coefficient (μm²/s)
        dt : float
            Time step (s)
        dim : int
            Number of dimensions (2 or 3)

        Returns
        -------
        trajectory : np.ndarray
            Shape (n_steps, dim) - positions at each time step
        """
        # Standard deviation of displacement per step
        sigma = np.sqrt(2 * D * dt)

        # Generate random steps
        steps = self.rng.normal(0, sigma, size=(n_steps - 1, dim))

        # Cumulative sum to get positions
        trajectory = np.zeros((n_steps, dim))
        trajectory[1:] = np.cumsum(steps, axis=0)

        return trajectory

    def confined_diffusion(self, n_steps: int, D: float = 1.0,
                          radius: float = 0.5, dt: float = 0.1,
                          dim: int = 2) -> np.ndarray:
        """
        Generate confined diffusion trajectory

        Particle diffuses within a circular/spherical confinement with
        reflecting boundaries.

        Parameters
        ----------
        n_steps : int
            Number of time steps
        D : float
            Diffusion coefficient (μm²/s)
        radius : float
            Confinement radius (μm)
        dt : float
            Time step (s)
        dim : int
            Dimensions (2 or 3)

        Returns
        -------
        trajectory : np.ndarray
            Shape (n_steps, dim)
        """
        sigma = np.sqrt(2 * D * dt)
        trajectory = np.zeros((n_steps, dim))

        for i in range(1, n_steps):
            # Propose new position
            step = self.rng.normal(0, sigma, size=dim)
            new_pos = trajectory[i-1] + step

            # Check if outside confinement
            distance = np.linalg.norm(new_pos)

            if distance > radius:
                # Reflect at boundary
                # Simple reflection: project back onto boundary
                new_pos = new_pos * (radius / distance)

            trajectory[i] = new_pos

        return trajectory

    def directed_motion(self, n_steps: int, velocity: float = 1.0,
                       D: float = 0.1, dt: float = 0.1,
                       angle: Optional[float] = None,
                       dim: int = 2) -> np.ndarray:
        """
        Generate directed motion with diffusion

        Combination of constant velocity and Brownian diffusion

        Parameters
        ----------
        n_steps : int
            Number of time steps
        velocity : float
            Drift velocity (μm/s)
        D : float
            Diffusion coefficient (μm²/s)
        dt : float
            Time step (s)
        angle : float, optional
            Direction angle (radians). If None, random direction
        dim : int
            Dimensions (2 or 3)

        Returns
        -------
        trajectory : np.ndarray
            Shape (n_steps, dim)
        """
        # Random direction if not specified
        if angle is None:
            if dim == 2:
                angle = self.rng.uniform(0, 2*np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])
            else:
                # Random direction on unit sphere
                direction = self.rng.normal(size=dim)
                direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([np.cos(angle), np.sin(angle)])

        # Drift component
        drift = velocity * dt * direction

        # Diffusion component
        sigma = np.sqrt(2 * D * dt)
        steps = self.rng.normal(0, sigma, size=(n_steps - 1, dim))

        # Combine drift and diffusion
        trajectory = np.zeros((n_steps, dim))
        for i in range(1, n_steps):
            trajectory[i] = trajectory[i-1] + drift + steps[i-1]

        return trajectory

    def fractional_brownian_motion(self, n_steps: int, H: float = 0.5,
                                   D: float = 1.0, dt: float = 0.1,
                                   dim: int = 2) -> np.ndarray:
        """
        Generate fractional Brownian motion (fBm)

        Parameters
        ----------
        n_steps : int
            Number of time steps
        H : float
            Hurst exponent (0 < H < 1)
            H = 0.5: Brownian motion
            H < 0.5: Subdiffusion (anti-persistent)
            H > 0.5: Superdiffusion (persistent)
        D : float
            Generalized diffusion coefficient
        dt : float
            Time step (s)
        dim : int
            Dimensions

        Returns
        -------
        trajectory : np.ndarray
            Shape (n_steps, dim)
        """
        # Generate correlated random steps using Davies-Harte method
        def generate_fgn(n, H):
            """Generate fractional Gaussian noise"""
            # Autocovariance function
            r = np.zeros(n)
            for k in range(n):
                r[k] = 0.5 * ((k+1)**(2*H) - 2*k**(2*H) + abs(k-1)**(2*H))

            # Circulant embedding
            c = np.concatenate([r, r[-2:0:-1]])

            # Eigenvalues via FFT
            eigenvalues = np.real(np.fft.fft(c))

            # Generate random phases
            w = self.rng.normal(size=len(eigenvalues))

            # Generate fGn
            fgn = np.real(np.fft.ifft(np.sqrt(eigenvalues) * np.fft.fft(w)))[:n]

            return fgn

        trajectory = np.zeros((n_steps, dim))

        # Generate for each dimension
        for d in range(dim):
            fgn = generate_fgn(n_steps - 1, H)
            # Scale by diffusion coefficient
            sigma = np.sqrt(D * dt**(2*H))
            trajectory[1:, d] = np.cumsum(fgn * sigma)

        return trajectory

    def continuous_time_random_walk(self, n_steps: int, D: float = 1.0,
                                   alpha: float = 0.5, dt: float = 0.1,
                                   dim: int = 2) -> np.ndarray:
        """
        Generate continuous-time random walk (CTRW) trajectory

        Subdiffusive behavior due to heavy-tailed waiting times

        Parameters
        ----------
        n_steps : int
            Number of observed time steps
        D : float
            Diffusion coefficient during movement
        alpha : float
            Waiting time exponent (0 < α < 1)
            Lower α = more trapping
        dt : float
            Observation time step
        dim : int
            Dimensions

        Returns
        -------
        trajectory : np.ndarray
            Shape (n_steps, dim)
        """
        # Generate waiting times from power-law distribution
        # ψ(τ) ~ τ^(-1-α)
        def power_law_waiting_times(n, alpha):
            u = self.rng.uniform(0, 1, n)
            # Inverse transform sampling
            return u**(-1/alpha)

        # Generate more steps than needed (will subsample)
        n_internal_steps = n_steps * 10
        waiting_times = power_law_waiting_times(n_internal_steps, alpha)

        # Normalize to match total observation time
        total_time = n_steps * dt
        waiting_times = waiting_times * (total_time / np.sum(waiting_times))

        # Generate positions
        sigma = np.sqrt(2 * D)
        steps = self.rng.normal(0, sigma, size=(n_internal_steps, dim))

        # Cumulative time and position
        cum_time = np.cumsum(waiting_times)
        cum_pos = np.cumsum(steps, axis=0)

        # Sample at regular intervals
        trajectory = np.zeros((n_steps, dim))
        for i in range(n_steps):
            t = i * dt
            idx = np.searchsorted(cum_time, t)
            if idx < len(cum_pos):
                trajectory[i] = cum_pos[idx]
            else:
                trajectory[i] = cum_pos[-1]

        return trajectory

    def generate_mixed_population(self, n_particles: int, n_steps: int,
                                  dt: float = 0.1,
                                  pixel_size: float = 0.065) -> pd.DataFrame:
        """
        Generate population with mixed mobility behaviors

        Parameters
        ----------
        n_particles : int
            Total number of particles
        n_steps : int
            Frames per trajectory
        dt : float
            Time step (s)
        pixel_size : float
            Pixel size (μm)

        Returns
        -------
        tracks : pd.DataFrame
            Synthetic trajectories in trackpy format
        """
        # Define population fractions
        fractions = {
            'brownian': 0.4,
            'confined': 0.25,
            'directed': 0.15,
            'subdiffusive': 0.15,
            'superdiffusive': 0.05
        }

        all_tracks = []
        particle_id = 0

        # Generate each type
        for mobility_type, fraction in fractions.items():
            n_type = int(n_particles * fraction)

            for _ in range(n_type):
                # Generate trajectory based on type
                if mobility_type == 'brownian':
                    D = self.rng.uniform(0.5, 2.0)
                    traj = self.brownian_motion(n_steps, D=D, dt=dt)

                elif mobility_type == 'confined':
                    D = self.rng.uniform(0.3, 1.0)
                    radius = self.rng.uniform(0.3, 0.8)
                    traj = self.confined_diffusion(n_steps, D=D, radius=radius, dt=dt)

                elif mobility_type == 'directed':
                    velocity = self.rng.uniform(0.5, 2.0)
                    D = self.rng.uniform(0.05, 0.2)
                    traj = self.directed_motion(n_steps, velocity=velocity, D=D, dt=dt)

                elif mobility_type == 'subdiffusive':
                    H = self.rng.uniform(0.2, 0.45)
                    D = self.rng.uniform(0.1, 0.5)
                    traj = self.fractional_brownian_motion(n_steps, H=H, D=D, dt=dt)

                elif mobility_type == 'superdiffusive':
                    H = self.rng.uniform(0.55, 0.8)
                    D = self.rng.uniform(1.0, 3.0)
                    traj = self.fractional_brownian_motion(n_steps, H=H, D=D, dt=dt)

                # Offset to different locations (in pixels)
                offset = self.rng.uniform(10, 500, size=2)
                traj_pixels = (traj / pixel_size) + offset

                # Create dataframe
                for frame_idx in range(n_steps):
                    all_tracks.append({
                        'particle': particle_id,
                        'frame': frame_idx,
                        'x': traj_pixels[frame_idx, 0],
                        'y': traj_pixels[frame_idx, 1],
                        'true_mobility': mobility_type
                    })

                particle_id += 1

        return pd.DataFrame(all_tracks)

    def generate_image_stack(self, tracks: pd.DataFrame,
                           image_size: Tuple[int, int] = (512, 512),
                           psf_width: float = 1.5,
                           intensity: float = 1000,
                           background: float = 100,
                           noise_level: float = 20) -> np.ndarray:
        """
        Generate synthetic microscopy images from trajectories

        Parameters
        ----------
        tracks : pd.DataFrame
            Trajectories (output from generate_mixed_population)
        image_size : tuple
            Image dimensions (height, width)
        psf_width : float
            Point spread function width (pixels)
        intensity : float
            Peak intensity of particles (photons)
        background : float
            Background level (photons)
        noise_level : float
            Poisson noise level

        Returns
        -------
        image_stack : np.ndarray
            Shape (n_frames, height, width)
        """
        n_frames = tracks['frame'].max() + 1
        image_stack = np.zeros((n_frames, *image_size))

        # Add background
        image_stack += background

        # Render each particle
        y_grid, x_grid = np.ogrid[0:image_size[0], 0:image_size[1]]

        for frame_idx in range(n_frames):
            frame_particles = tracks[tracks['frame'] == frame_idx]

            for _, particle in frame_particles.iterrows():
                x0, y0 = particle['x'], particle['y']

                # Skip if outside image
                if (x0 < 0 or x0 >= image_size[1] or
                    y0 < 0 or y0 >= image_size[0]):
                    continue

                # Gaussian PSF
                gaussian = np.exp(-((x_grid - x0)**2 + (y_grid - y0)**2) /
                                (2 * psf_width**2))

                # Add to image
                image_stack[frame_idx] += intensity * gaussian

        # Add Poisson noise
        image_stack = self.rng.poisson(image_stack).astype(np.float32)

        # Add Gaussian read noise
        image_stack += self.rng.normal(0, noise_level, image_stack.shape)
        image_stack = np.maximum(image_stack, 0)  # No negative values

        return image_stack


class ValidationSuite:
    """
    Validate tracking and analysis algorithms using simulated data
    """

    def __init__(self):
        self.simulator = TrajectorySimulator(seed=42)

    def test_detection_accuracy(self, tracker, n_particles: int = 50,
                               n_frames: int = 100) -> Dict:
        """
        Test particle detection accuracy

        Returns
        -------
        results : dict
            Detection performance metrics
        """
        from particle_tracker import ParticleTracker

        # Generate ground truth
        tracks_gt = self.simulator.generate_mixed_population(
            n_particles, n_frames
        )

        # Generate images
        images = self.simulator.generate_image_stack(tracks_gt)

        # Detect particles
        particles_detected = tracker.detect_particles(images, diameter=9)

        # Calculate metrics per frame
        recall_per_frame = []
        precision_per_frame = []

        for frame_idx in range(n_frames):
            gt_frame = tracks_gt[tracks_gt['frame'] == frame_idx]
            det_frame = particles_detected[particles_detected['frame'] == frame_idx]

            n_true = len(gt_frame)
            n_detected = len(det_frame)

            # Count true positives (within 2 pixels)
            tp = 0
            for _, gt_particle in gt_frame.iterrows():
                distances = np.sqrt(
                    (det_frame['x'] - gt_particle['x'])**2 +
                    (det_frame['y'] - gt_particle['y'])**2
                )
                if len(distances) > 0 and distances.min() < 2.0:
                    tp += 1

            recall = tp / n_true if n_true > 0 else 0
            precision = tp / n_detected if n_detected > 0 else 0

            recall_per_frame.append(recall)
            precision_per_frame.append(precision)

        return {
            'mean_recall': np.mean(recall_per_frame),
            'mean_precision': np.mean(precision_per_frame),
            'f1_score': 2 * np.mean(recall_per_frame) * np.mean(precision_per_frame) /
                       (np.mean(recall_per_frame) + np.mean(precision_per_frame))
        }

    def test_diffusion_estimation(self, analyzer,
                                 true_D: float = 1.0,
                                 n_trajectories: int = 100) -> Dict:
        """
        Test diffusion coefficient estimation accuracy

        Returns
        -------
        results : dict
            Estimation errors and statistics
        """
        estimated_D = []

        for _ in range(n_trajectories):
            # Generate Brownian trajectory
            traj = self.simulator.brownian_motion(100, D=true_D, dt=0.1)

            # Convert to DataFrame
            traj_df = pd.DataFrame({
                'particle': 0,
                'frame': np.arange(len(traj)),
                'x': traj[:, 0],
                'y': traj[:, 1]
            })

            # Estimate D
            D_est = analyzer.calculate_diffusion_coefficients(
                traj_df, pixel_size=1.0, fps=10.0
            )[0]

            estimated_D.append(D_est)

        estimated_D = np.array(estimated_D)
        estimated_D = estimated_D[~np.isnan(estimated_D)]

        return {
            'true_D': true_D,
            'mean_estimated_D': np.mean(estimated_D),
            'median_estimated_D': np.median(estimated_D),
            'std_estimated_D': np.std(estimated_D),
            'relative_error': np.abs(np.mean(estimated_D) - true_D) / true_D,
            'rmse': np.sqrt(np.mean((estimated_D - true_D)**2))
        }

    def test_mobility_classification(self, analyzer,
                                    n_per_class: int = 50) -> pd.DataFrame:
        """
        Test mobility classification accuracy

        Returns
        -------
        confusion_matrix : pd.DataFrame
            Confusion matrix of classifications
        """
        # Generate test data
        tracks = self.simulator.generate_mixed_population(
            n_per_class * 5, 100
        )

        # Classify
        predicted_classes = analyzer.classify_mobility(tracks)

        # Get true labels
        true_labels = []
        for particle_id in tracks['particle'].unique():
            true_labels.append(
                tracks[tracks['particle'] == particle_id]['true_mobility'].iloc[0]
            )

        # Create confusion matrix
        from collections import defaultdict
        confusion = defaultdict(lambda: defaultdict(int))

        for true, pred in zip(true_labels, predicted_classes):
            confusion[true][pred] += 1

        # Convert to DataFrame
        classes = ['brownian', 'confined', 'directed', 'subdiffusive', 'superdiffusive']
        conf_df = pd.DataFrame(0, index=classes, columns=classes)

        for true_class in confusion:
            for pred_class in confusion[true_class]:
                if true_class in classes and pred_class in classes:
                    conf_df.loc[true_class, pred_class] = confusion[true_class][pred_class]

        return conf_df


# Example usage
if __name__ == '__main__':
    # Create simulator
    simulator = TrajectorySimulator(seed=42)

    # Generate synthetic data
    print("Generating synthetic trajectories...")
    tracks = simulator.generate_mixed_population(
        n_particles=100,
        n_steps=200,
        dt=0.1,
        pixel_size=0.065
    )

    print(f"Generated {tracks['particle'].nunique()} trajectories")
    print(f"Mobility distribution:")
    print(tracks.groupby('particle')['true_mobility'].first().value_counts())

    # Generate images
    print("\nGenerating synthetic images...")
    images = simulator.generate_image_stack(tracks, intensity=500)

    print(f"Image stack shape: {images.shape}")
    print(f"Mean intensity: {images.mean():.1f}")

    # Save
    from skimage import io
    io.imsave('synthetic_trajectories.tif', images.astype(np.uint16))
    tracks.to_csv('synthetic_tracks_ground_truth.csv', index=False)

    print("\nSynthetic data saved!")
    print("- synthetic_trajectories.tif")
    print("- synthetic_tracks_ground_truth.csv")
