"""
Complete Analysis Workflow Example
===================================

End-to-end example of analyzing particle tracking data with NanoPyx.

This script demonstrates:
1. Loading microscopy data
2. Image preprocessing with NanoPyx
3. Particle detection and tracking
4. Mobility analysis and classification
5. Visualization and export
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from nanopyx_processor import NanoPyxProcessor
from particle_tracker import ParticleTracker
from mobility_analyzer import MobilityAnalyzer

# For generating test data
from trajectory_simulator import TrajectorySimulator


def example_1_basic_workflow():
    """
    Example 1: Basic workflow with synthetic data
    """
    print("=" * 60)
    print("Example 1: Basic Workflow")
    print("=" * 60)
    
    # Step 1: Generate or load data
    print("\n[1/6] Generating synthetic data...")
    simulator = TrajectorySimulator(seed=42)
    
    # Create mixed population
    tracks_gt = simulator.generate_mixed_population(
        n_particles=50,
        n_steps=100,
        dt=0.1,  # 10 Hz acquisition
        pixel_size=0.065  # 65 nm pixels
    )
    
    # Generate microscopy images
    images = simulator.generate_image_stack(
        tracks_gt,
        image_size=(512, 512),
        psf_width=1.5,
        intensity=800,
        background=50,
        noise_level=15
    )
    
    print(f"   Created {images.shape[0]} frames of size {images.shape[1]}x{images.shape[2]}")
    print(f"   Ground truth: {tracks_gt['particle'].nunique()} particles")
    
    # Step 2: Image preprocessing (optional with NanoPyx)
    print("\n[2/6] Preprocessing images...")
    processor = NanoPyxProcessor()
    
    # Apply drift correction
    images_corrected = processor.drift_correction(images)
    print("   ✓ Drift correction applied")
    
    # Step 3: Particle detection
    print("\n[3/6] Detecting particles...")
    tracker = ParticleTracker()
    
    particles = tracker.detect_particles(
        images_corrected,
        diameter=9,
        minmass=100,
        percentile=60
    )
    
    print(f"   Detected {len(particles)} particle localizations")
    print(f"   Average per frame: {len(particles) / images.shape[0]:.1f}")
    
    # Step 4: Link particles into tracks
    print("\n[4/6] Linking particles into trajectories...")
    tracks = tracker.link_particles(
        particles,
        search_range=5.0,
        memory=3
    )
    
    print(f"   Linked into {tracks['particle'].nunique()} trajectories")
    
    # Filter short tracks
    tracks = tracker.filter_tracks(tracks, min_length=15)
    print(f"   After filtering: {tracks['particle'].nunique()} trajectories")
    
    # Step 5: Mobility analysis
    print("\n[5/6] Analyzing particle mobility...")
    analyzer = MobilityAnalyzer()
    
    # Get complete summary
    summary = analyzer.get_mobility_summary(
        tracks,
        pixel_size=0.065,
        fps=10.0
    )
    
    # Calculate MSD
    msd = analyzer.calculate_msd(
        tracks,
        pixel_size=0.065,
        fps=10.0,
        max_lagtime=30
    )
    
    # Print results
    print("\n   Mobility Classification Results:")
    print("   " + "-" * 40)
    for cls, count in summary['mobility_class'].value_counts().items():
        pct = 100 * count / len(summary)
        print(f"   {cls.capitalize():15s}: {count:3d} ({pct:5.1f}%)")
    
    print("\n   Diffusion Coefficients:")
    print("   " + "-" * 40)
    D_values = summary['diffusion_coefficient'].dropna()
    print(f"   Mean:   {D_values.mean():.3f} μm²/s")
    print(f"   Median: {D_values.median():.3f} μm²/s")
    print(f"   Range:  [{D_values.min():.3f}, {D_values.max():.3f}] μm²/s")
    
    # Step 6: Visualization and export
    print("\n[6/6] Creating visualizations...")
    
    # Create output directory
    output_dir = Path("output_example1")
    output_dir.mkdir(exist_ok=True)
    
    # Plot MSD
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(msd.index, msd['msd'], 'b-', linewidth=2)
    ax.set_xlabel('Time lag (s)', fontsize=12)
    ax.set_ylabel('MSD (μm²)', fontsize=12)
    ax.set_title('Ensemble Mean Square Displacement', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'msd_plot.png', dpi=150)
    print(f"   ✓ Saved {output_dir}/msd_plot.png")
    
    # Plot diffusion coefficient histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        'brownian': 'blue', 'confined': 'green', 'directed': 'red',
        'subdiffusive': 'orange', 'superdiffusive': 'purple'
    }
    
    for cls in summary['mobility_class'].unique():
        if cls in colors:
            data = summary[summary['mobility_class'] == cls]['diffusion_coefficient']
            data = data.dropna()
            ax.hist(data, bins=20, alpha=0.6, label=cls.capitalize(), color=colors[cls])
    
    ax.set_xlabel('Diffusion Coefficient (μm²/s)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Diffusion Coefficients', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'diffusion_histogram.png', dpi=150)
    print(f"   ✓ Saved {output_dir}/diffusion_histogram.png")
    
    # Plot trajectories
    fig, ax = plt.subplots(figsize=(10, 10))
    tracker.plot_trajectories(tracks, colorby='particle', ax=ax)
    plt.savefig(output_dir / 'trajectories.png', dpi=150)
    print(f"   ✓ Saved {output_dir}/trajectories.png")
    
    # Export data
    summary.to_csv(output_dir / 'mobility_summary.csv', index=False)
    tracks.to_csv(output_dir / 'tracks.csv', index=False)
    msd.to_csv(output_dir / 'msd.csv')
    
    print(f"   ✓ Exported data to {output_dir}/")
    
    print("\n" + "=" * 60)
    print("Example 1 Complete!")
    print("=" * 60)


def example_2_real_data_workflow():
    """
    Example 2: Workflow for real experimental data
    """
    print("\n" + "=" * 60)
    print("Example 2: Real Data Workflow")
    print("=" * 60)
    
    # This example shows how to load and process real microscopy data
    
    print("\n[Info] This example requires real microscopy data.")
    print("       Replace 'your_data.tif' with your actual file.")
    
    # Uncomment and modify for your data:
    """
    from skimage import io
    
    # Load your data
    images = io.imread('your_data.tif')
    print(f"Loaded {images.shape}")
    
    # Configure for your experimental conditions
    PIXEL_SIZE = 0.065  # μm (100x/1.45 NA objective)
    FRAME_RATE = 10.0   # Hz
    
    # Initialize processors
    processor = NanoPyxProcessor()
    tracker = ParticleTracker()
    analyzer = MobilityAnalyzer()
    
    # Full processing pipeline
    print("Drift correction...")
    corrected = processor.drift_correction(images)
    
    print("Detecting particles...")
    particles = tracker.detect_particles(
        corrected,
        diameter=11,      # Adjust for your particle size
        minmass=150,      # Adjust threshold
        percentile=64
    )
    
    print("Linking trajectories...")
    tracks = tracker.link_particles(
        particles,
        search_range=5.0,  # Adjust for your dynamics
        memory=3
    )
    
    tracks = tracker.filter_tracks(tracks, min_length=20)
    
    print("Analyzing mobility...")
    summary = analyzer.get_mobility_summary(
        tracks,
        pixel_size=PIXEL_SIZE,
        fps=FRAME_RATE
    )
    
    # Export results
    summary.to_csv('results/mobility_analysis.csv', index=False)
    tracks.to_csv('results/tracks.csv', index=False)
    
    print("Analysis complete!")
    """
    
    print("\n   Template code shown above (currently commented out)")
    print("   Uncomment and adapt for your specific data")


def example_3_batch_processing():
    """
    Example 3: Batch process multiple files
    """
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)
    
    print("\n[Info] Example of batch processing multiple datasets")
    
    # Template for batch processing
    code = """
    from pathlib import Path
    from skimage import io
    import pandas as pd
    
    # Configuration
    DATA_DIR = Path('data')
    OUTPUT_DIR = Path('results')
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    PIXEL_SIZE = 0.065
    FRAME_RATE = 10.0
    
    # Initialize once
    processor = NanoPyxProcessor()
    tracker = ParticleTracker()
    analyzer = MobilityAnalyzer()
    
    # Process all TIFF files
    all_results = []
    
    for data_file in DATA_DIR.glob('*.tif'):
        print(f"Processing {data_file.name}...")
        
        try:
            # Load
            images = io.imread(data_file)
            
            # Process
            corrected = processor.drift_correction(images)
            particles = tracker.detect_particles(corrected, diameter=11)
            tracks = tracker.link_particles(particles, search_range=5.0)
            tracks = tracker.filter_tracks(tracks, min_length=15)
            
            # Analyze
            summary = analyzer.get_mobility_summary(
                tracks, 
                pixel_size=PIXEL_SIZE, 
                fps=FRAME_RATE
            )
            
            # Add file identifier
            summary['file'] = data_file.stem
            all_results.append(summary)
            
            # Save individual results
            output_file = OUTPUT_DIR / f"{data_file.stem}_analysis.csv"
            summary.to_csv(output_file, index=False)
            
            print(f"  ✓ Analyzed {len(summary)} tracks")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / 'combined_analysis.csv', index=False)
        
        print(f"\\nBatch processing complete!")
        print(f"Total tracks analyzed: {len(combined)}")
        print(f"Results saved to {OUTPUT_DIR}/")
    """
    
    print(code)


def example_4_advanced_analysis():
    """
    Example 4: Advanced analysis - comparing conditions
    """
    print("\n" + "=" * 60)
    print("Example 4: Advanced Analysis")
    print("=" * 60)
    
    print("\n[Info] Comparing different experimental conditions")
    
    # Generate two populations with different characteristics
    simulator = TrajectorySimulator(seed=42)
    
    # Condition 1: Normal (control)
    print("\nGenerating control condition...")
    tracks_control = simulator.generate_mixed_population(30, 100)
    
    # Condition 2: Treatment (more confined)
    print("Generating treatment condition...")
    # Modify to have more confined motion
    tracks_treatment = simulator.generate_mixed_population(30, 100)
    # (In real analysis, these would be separate datasets)
    
    # Analyze both
    analyzer = MobilityAnalyzer()
    
    summary_control = analyzer.get_mobility_summary(
        tracks_control, pixel_size=0.065, fps=10.0
    )
    summary_treatment = analyzer.get_mobility_summary(
        tracks_treatment, pixel_size=0.065, fps=10.0
    )
    
    # Compare
    print("\n" + "=" * 60)
    print("Comparison: Control vs Treatment")
    print("=" * 60)
    
    print("\nMobility Class Distribution:")
    print("-" * 60)
    print(f"{'Class':<15} {'Control':>15} {'Treatment':>15}")
    print("-" * 60)
    
    classes = ['brownian', 'confined', 'directed', 'subdiffusive']
    for cls in classes:
        n_ctrl = (summary_control['mobility_class'] == cls).sum()
        n_treat = (summary_treatment['mobility_class'] == cls).sum()
        pct_ctrl = 100 * n_ctrl / len(summary_control)
        pct_treat = 100 * n_treat / len(summary_treatment)
        print(f"{cls.capitalize():<15} {n_ctrl:3d} ({pct_ctrl:5.1f}%) {n_treat:3d} ({pct_treat:5.1f}%)")
    
    # Statistical comparison
    print("\nDiffusion Coefficients:")
    print("-" * 60)
    D_ctrl = summary_control['diffusion_coefficient'].dropna()
    D_treat = summary_treatment['diffusion_coefficient'].dropna()
    
    print(f"Control:   {D_ctrl.mean():.3f} ± {D_ctrl.std():.3f} μm²/s")
    print(f"Treatment: {D_treat.mean():.3f} ± {D_treat.std():.3f} μm²/s")
    
    # Simple t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(D_ctrl, D_treat)
    print(f"\nt-test: t={t_stat:.3f}, p={p_value:.4f}")
    
    if p_value < 0.05:
        print("Significant difference detected!")
    else:
        print("No significant difference")


def example_5_quality_control():
    """
    Example 5: Quality control and validation
    """
    print("\n" + "=" * 60)
    print("Example 5: Quality Control")
    print("=" * 60)
    
    # Generate test data
    simulator = TrajectorySimulator(seed=42)
    tracks_gt = simulator.generate_mixed_population(30, 100)
    images = simulator.generate_image_stack(tracks_gt)
    
    # Process
    tracker = ParticleTracker()
    particles = tracker.detect_particles(images, diameter=9, minmass=100)
    tracks = tracker.link_particles(particles, search_range=5.0)
    tracks = tracker.filter_tracks(tracks, min_length=10)
    
    print("\n[Quality Control Checks]")
    print("-" * 60)
    
    # 1. Detection rate
    expected_per_frame = tracks_gt['particle'].nunique()
    detected_per_frame = len(particles) / images.shape[0]
    print(f"1. Detection rate:")
    print(f"   Expected: ~{expected_per_frame} particles/frame")
    print(f"   Detected: {detected_per_frame:.1f} particles/frame")
    
    # 2. Track length distribution
    track_lengths = tracks.groupby('particle').size()
    print(f"\n2. Track length statistics:")
    print(f"   Mean:   {track_lengths.mean():.1f} frames")
    print(f"   Median: {track_lengths.median():.1f} frames")
    print(f"   Min:    {track_lengths.min()} frames")
    print(f"   Max:    {track_lengths.max()} frames")
    
    # 3. Localization precision check
    print(f"\n3. Localization precision:")
    # Short tracks are red flag
    short_tracks = (track_lengths < 10).sum()
    pct_short = 100 * short_tracks / len(track_lengths)
    print(f"   Short tracks (<10 frames): {short_tracks} ({pct_short:.1f}%)")
    if pct_short > 30:
        print("   ⚠ Warning: Many short tracks - check detection parameters")
    else:
        print("   ✓ OK")
    
    # 4. MSD quality
    analyzer = MobilityAnalyzer()
    summary = analyzer.get_mobility_summary(tracks, pixel_size=0.065, fps=10.0)
    
    print(f"\n4. Diffusion coefficient sanity check:")
    D_values = summary['diffusion_coefficient'].dropna()
    if len(D_values) > 0:
        # Check for negative or unrealistic values
        n_negative = (D_values < 0).sum()
        n_too_large = (D_values > 100).sum()
        
        print(f"   Negative D: {n_negative}")
        print(f"   D > 100 μm²/s: {n_too_large}")
        
        if n_negative > 0:
            print("   ⚠ Warning: Negative diffusion coefficients detected")
            print("      → Check localization precision")
            print("      → Increase track length threshold")
        
        if n_too_large > len(D_values) * 0.1:
            print("   ⚠ Warning: Many unusually large D values")
            print("      → Check pixel size calibration")
            print("      → Check for linking errors")
        
        if n_negative == 0 and n_too_large < len(D_values) * 0.05:
            print("   ✓ All values within reasonable range")
    
    print("\n" + "=" * 60)


def main():
    """
    Run all examples
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "NanoPyx Particle Tracking Examples")
    print("=" * 70)
    
    # Example 1: Basic workflow
    example_1_basic_workflow()
    
    # Example 2: Real data template
    example_2_real_data_workflow()
    
    # Example 3: Batch processing
    example_3_batch_processing()
    
    # Example 4: Advanced analysis
    example_4_advanced_analysis()
    
    # Example 5: Quality control
    example_5_quality_control()
    
    print("\n" + "=" * 70)
    print(" " * 20 + "All Examples Complete!")
    print("=" * 70)
    print("\nCheck the 'output_example1/' directory for results from Example 1")
    print("\nFor more information, see the documentation and README")


if __name__ == '__main__':
    main()
