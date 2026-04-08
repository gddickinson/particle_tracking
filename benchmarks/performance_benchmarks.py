"""
Performance Benchmarking Suite
===============================

Comprehensive benchmarks for all components:
- NanoPyx Liquid Engine performance
- Particle detection speed
- Tracking algorithm efficiency
- Mobility analysis computational cost

Use this to optimize parameters for your hardware and validate system performance.
"""

import numpy as np
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

from nanopyx_processor import NanoPyxProcessor
from particle_tracker import ParticleTracker
from mobility_analyzer import MobilityAnalyzer
from trajectory_simulator import TrajectorySimulator


class PerformanceBenchmark:
    """
    Benchmark suite for particle tracking system
    """
    
    def __init__(self):
        self.results = []
        self.processor = NanoPyxProcessor()
        self.tracker = ParticleTracker()
        self.analyzer = MobilityAnalyzer()
        self.simulator = TrajectorySimulator(seed=42)
    
    def benchmark_nanopyx_drift_correction(self,
                                          image_sizes: List[tuple] = None,
                                          n_frames: int = 100) -> pd.DataFrame:
        """
        Benchmark NanoPyx drift correction performance
        """
        if image_sizes is None:
            image_sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
        
        print("\n" + "=" * 60)
        print("Benchmarking NanoPyx Drift Correction")
        print("=" * 60)
        
        results = []
        
        for size in image_sizes:
            # Generate test data
            print(f"\nTesting {size[0]}x{size[1]} images...")
            stack = np.random.randn(n_frames, *size).astype(np.float32) * 100 + 1000
            
            # Benchmark
            start = time.time()
            corrected = self.processor.drift_correction(stack)
            elapsed = time.time() - start
            
            # Calculate metrics
            fps = n_frames / elapsed
            mpixels_per_sec = (n_frames * size[0] * size[1]) / elapsed / 1e6
            
            result = {
                'task': 'drift_correction',
                'n_frames': n_frames,
                'image_size': f"{size[0]}x{size[1]}",
                'total_pixels': n_frames * size[0] * size[1],
                'time_seconds': elapsed,
                'frames_per_second': fps,
                'megapixels_per_second': mpixels_per_sec
            }
            
            results.append(result)
            
            print(f"   Time: {elapsed:.2f} s")
            print(f"   Speed: {fps:.1f} frames/s, {mpixels_per_sec:.1f} Mpixels/s")
        
        return pd.DataFrame(results)
    
    def benchmark_particle_detection(self,
                                     particle_densities: List[float] = None,
                                     image_size: tuple = (512, 512),
                                     n_frames: int = 100) -> pd.DataFrame:
        """
        Benchmark particle detection speed vs. particle density
        """
        if particle_densities is None:
            particle_densities = [10, 50, 100, 200, 500]
        
        print("\n" + "=" * 60)
        print("Benchmarking Particle Detection")
        print("=" * 60)
        
        results = []
        
        for n_particles in particle_densities:
            print(f"\nTesting {n_particles} particles/frame...")
            
            # Generate synthetic data
            tracks = self.simulator.generate_mixed_population(
                n_particles, n_frames, pixel_size=0.065
            )
            images = self.simulator.generate_image_stack(
                tracks, image_size=image_size
            )
            
            # Benchmark detection
            start = time.time()
            particles = self.tracker.detect_particles(
                images,
                diameter=11,
                minmass=100
            )
            elapsed = time.time() - start
            
            # Metrics
            fps = n_frames / elapsed
            particles_per_sec = len(particles) / elapsed
            
            result = {
                'task': 'detection',
                'n_frames': n_frames,
                'particles_per_frame': n_particles,
                'total_detections': len(particles),
                'time_seconds': elapsed,
                'frames_per_second': fps,
                'detections_per_second': particles_per_sec
            }
            
            results.append(result)
            
            print(f"   Detected: {len(particles)} particles")
            print(f"   Time: {elapsed:.2f} s ({fps:.1f} frames/s)")
            print(f"   Speed: {particles_per_sec:.0f} detections/s")
        
        return pd.DataFrame(results)
    
    def benchmark_particle_linking(self,
                                   n_particles_list: List[int] = None,
                                   n_frames: int = 100) -> pd.DataFrame:
        """
        Benchmark particle linking speed vs. number of particles
        """
        if n_particles_list is None:
            n_particles_list = [10, 50, 100, 200, 500]
        
        print("\n" + "=" * 60)
        print("Benchmarking Particle Linking")
        print("=" * 60)
        
        results = []
        
        for n_particles in n_particles_list:
            print(f"\nTesting {n_particles} particles...")
            
            # Generate synthetic detections
            tracks = self.simulator.generate_mixed_population(
                n_particles, n_frames
            )
            
            # Simulate as if they were detections (no linking yet)
            particles = tracks[['frame', 'x', 'y']].copy()
            particles['mass'] = 1000.0
            particles['size'] = 1.5
            particles['ecc'] = 0.1
            
            # Benchmark linking
            start = time.time()
            linked = self.tracker.link_particles(
                particles,
                search_range=5.0,
                memory=3
            )
            elapsed = time.time() - start
            
            # Metrics
            n_tracks = linked['particle'].nunique()
            links_per_sec = len(particles) / elapsed
            
            result = {
                'task': 'linking',
                'n_particles': n_particles,
                'n_frames': n_frames,
                'n_localizations': len(particles),
                'n_tracks': n_tracks,
                'time_seconds': elapsed,
                'localizations_per_second': links_per_sec
            }
            
            results.append(result)
            
            print(f"   Linked to: {n_tracks} tracks")
            print(f"   Time: {elapsed:.2f} s")
            print(f"   Speed: {links_per_sec:.0f} localizations/s")
        
        return pd.DataFrame(results)
    
    def benchmark_msd_calculation(self,
                                  track_lengths: List[int] = None,
                                  n_tracks: int = 100) -> pd.DataFrame:
        """
        Benchmark MSD calculation speed vs. track length
        """
        if track_lengths is None:
            track_lengths = [10, 50, 100, 200, 500, 1000]
        
        print("\n" + "=" * 60)
        print("Benchmarking MSD Calculation")
        print("=" * 60)
        
        results = []
        
        for length in track_lengths:
            print(f"\nTesting tracks of length {length}...")
            
            # Generate synthetic tracks
            all_tracks = []
            for i in range(n_tracks):
                traj = self.simulator.brownian_motion(length, D=1.0, dt=0.1)
                for j in range(length):
                    all_tracks.append({
                        'particle': i,
                        'frame': j,
                        'x': traj[j, 0],
                        'y': traj[j, 1]
                    })
            
            tracks_df = pd.DataFrame(all_tracks)
            
            # Benchmark MSD calculation
            start = time.time()
            msd = self.analyzer.calculate_msd(
                tracks_df,
                pixel_size=0.065,
                fps=10.0,
                max_lagtime=min(50, length // 2)
            )
            elapsed = time.time() - start
            
            # Metrics
            msds_per_sec = n_tracks / elapsed
            
            result = {
                'task': 'msd_calculation',
                'n_tracks': n_tracks,
                'track_length': length,
                'total_points': len(tracks_df),
                'time_seconds': elapsed,
                'tracks_per_second': msds_per_sec
            }
            
            results.append(result)
            
            print(f"   Time: {elapsed:.3f} s")
            print(f"   Speed: {msds_per_sec:.1f} tracks/s")
        
        return pd.DataFrame(results)
    
    def benchmark_mobility_classification(self,
                                         n_tracks_list: List[int] = None,
                                         track_length: int = 100) -> pd.DataFrame:
        """
        Benchmark mobility classification speed
        """
        if n_tracks_list is None:
            n_tracks_list = [10, 50, 100, 500, 1000]
        
        print("\n" + "=" * 60)
        print("Benchmarking Mobility Classification")
        print("=" * 60)
        
        results = []
        
        for n_tracks in n_tracks_list:
            print(f"\nTesting {n_tracks} tracks...")
            
            # Generate tracks
            tracks = self.simulator.generate_mixed_population(
                n_tracks, track_length
            )
            
            # Benchmark classification
            start = time.time()
            classes = self.analyzer.classify_mobility(
                tracks,
                pixel_size=0.065,
                fps=10.0
            )
            elapsed = time.time() - start
            
            # Metrics
            tracks_per_sec = n_tracks / elapsed
            
            result = {
                'task': 'classification',
                'n_tracks': n_tracks,
                'track_length': track_length,
                'time_seconds': elapsed,
                'tracks_per_second': tracks_per_sec
            }
            
            results.append(result)
            
            print(f"   Time: {elapsed:.3f} s")
            print(f"   Speed: {tracks_per_sec:.1f} tracks/s")
        
        return pd.DataFrame(results)
    
    def run_full_benchmark_suite(self) -> Dict[str, pd.DataFrame]:
        """
        Run complete benchmark suite
        """
        print("\n" + "=" * 70)
        print(" " * 20 + "FULL BENCHMARK SUITE")
        print("=" * 70)
        
        results = {}
        
        # 1. Drift correction
        results['drift_correction'] = self.benchmark_nanopyx_drift_correction(
            image_sizes=[(512, 512), (1024, 1024)],
            n_frames=50
        )
        
        # 2. Detection
        results['detection'] = self.benchmark_particle_detection(
            particle_densities=[50, 100, 200],
            n_frames=50
        )
        
        # 3. Linking
        results['linking'] = self.benchmark_particle_linking(
            n_particles_list=[50, 100, 200],
            n_frames=50
        )
        
        # 4. MSD calculation
        results['msd'] = self.benchmark_msd_calculation(
            track_lengths=[50, 100, 200],
            n_tracks=50
        )
        
        # 5. Classification
        results['classification'] = self.benchmark_mobility_classification(
            n_tracks_list=[50, 100, 500],
            track_length=100
        )
        
        return results
    
    def generate_benchmark_report(self, results: Dict[str, pd.DataFrame],
                                 output_dir: Path = None):
        """
        Generate comprehensive benchmark report
        """
        if output_dir is None:
            output_dir = Path('benchmark_results')
        output_dir.mkdir(exist_ok=True)
        
        print("\n" + "=" * 70)
        print(" " * 20 + "GENERATING REPORT")
        print("=" * 70)
        
        # Save all results
        for task, df in results.items():
            csv_path = output_dir / f'{task}_benchmark.csv'
            df.to_csv(csv_path, index=False)
            print(f"✓ Saved {csv_path}")
        
        # Create summary plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Performance Benchmarks', fontsize=16, fontweight='bold')
        
        # Plot 1: Drift correction
        ax = axes[0, 0]
        df = results['drift_correction']
        ax.bar(range(len(df)), df['frames_per_second'])
        ax.set_xlabel('Image Size')
        ax.set_ylabel('Frames/second')
        ax.set_title('Drift Correction Speed')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['image_size'], rotation=45)
        
        # Plot 2: Detection
        ax = axes[0, 1]
        df = results['detection']
        ax.plot(df['particles_per_frame'], df['frames_per_second'], 'o-', linewidth=2)
        ax.set_xlabel('Particles per Frame')
        ax.set_ylabel('Frames/second')
        ax.set_title('Detection Speed')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Linking
        ax = axes[0, 2]
        df = results['linking']
        ax.plot(df['n_particles'], df['localizations_per_second'], 's-', linewidth=2, color='green')
        ax.set_xlabel('Number of Particles')
        ax.set_ylabel('Localizations/second')
        ax.set_title('Linking Speed')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: MSD
        ax = axes[1, 0]
        df = results['msd']
        ax.plot(df['track_length'], df['tracks_per_second'], '^-', linewidth=2, color='orange')
        ax.set_xlabel('Track Length (frames)')
        ax.set_ylabel('Tracks/second')
        ax.set_title('MSD Calculation Speed')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Classification
        ax = axes[1, 1]
        df = results['classification']
        ax.plot(df['n_tracks'], df['tracks_per_second'], 'd-', linewidth=2, color='red')
        ax.set_xlabel('Number of Tracks')
        ax.set_ylabel('Tracks/second')
        ax.set_title('Classification Speed')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Summary comparison
        ax = axes[1, 2]
        tasks = ['Detection', 'Linking', 'MSD', 'Classification']
        # Take representative speeds from each benchmark
        speeds = [
            results['detection']['frames_per_second'].mean(),
            results['linking']['localizations_per_second'].mean() / 100,  # Normalize
            results['msd']['tracks_per_second'].mean(),
            results['classification']['tracks_per_second'].mean()
        ]
        ax.barh(tasks, speeds, color=['blue', 'green', 'orange', 'red'])
        ax.set_xlabel('Relative Speed (normalized)')
        ax.set_title('Task Comparison')
        
        plt.tight_layout()
        plot_path = output_dir / 'benchmark_summary.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved {plot_path}")
        
        # Generate text report
        report_path = output_dir / 'benchmark_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(" " * 20 + "PERFORMANCE BENCHMARK REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            for task, df in results.items():
                f.write(f"\n{task.upper().replace('_', ' ')}\n")
                f.write("-" * 70 + "\n")
                f.write(df.to_string(index=False))
                f.write("\n\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"\nDrift Correction: {results['drift_correction']['frames_per_second'].mean():.1f} frames/s avg\n")
            f.write(f"Detection: {results['detection']['detections_per_second'].mean():.0f} detections/s avg\n")
            f.write(f"Linking: {results['linking']['localizations_per_second'].mean():.0f} links/s avg\n")
            f.write(f"MSD Calculation: {results['msd']['tracks_per_second'].mean():.1f} tracks/s avg\n")
            f.write(f"Classification: {results['classification']['tracks_per_second'].mean():.1f} tracks/s avg\n")
        
        print(f"✓ Saved {report_path}")
        
        print(f"\n✓ Complete benchmark report in: {output_dir}/")


def main():
    """
    Run benchmarks
    """
    import sys
    
    benchmark = PerformanceBenchmark()
    
    # Check for quick vs full benchmark
    if '--quick' in sys.argv:
        print("Running quick benchmarks...")
        results = {
            'detection': benchmark.benchmark_particle_detection(
                particle_densities=[50, 100],
                n_frames=20
            ),
            'linking': benchmark.benchmark_particle_linking(
                n_particles_list=[50, 100],
                n_frames=20
            )
        }
    else:
        print("Running full benchmark suite...")
        print("(Use --quick flag for faster testing)")
        results = benchmark.run_full_benchmark_suite()
    
    # Generate report
    benchmark.generate_benchmark_report(results)
    
    print("\n" + "=" * 70)
    print(" " * 25 + "BENCHMARKS COMPLETE!")
    print("=" * 70)
    print("\nResults saved to: benchmark_results/")
    print("\nCheck:")
    print("  - benchmark_results/*.csv for raw data")
    print("  - benchmark_results/benchmark_summary.png for plots")
    print("  - benchmark_results/benchmark_report.txt for full report")


if __name__ == '__main__':
    main()
