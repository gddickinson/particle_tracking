"""
Advanced Particle Tracking System with NanoPyx Integration
===========================================================

This application combines NanoPyx's high-performance image processing with
modern particle tracking techniques to analyze fluorescently-tagged proteins
with mixed mobility behaviors.

Features:
- NanoPyx-powered image enhancement (eSRRF, drift correction)
- Trackpy-based particle detection and linking
- Mobility classification (Brownian, confined, directed, anomalous)
- Real-time PyQt6 visualization
- MSD and diffusion coefficient analysis
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QPushButton, QLabel, QFileDialog,
                              QTabWidget, QSpinBox, QDoubleSpinBox, QGroupBox,
                              QFormLayout, QProgressBar, QTextEdit, QComboBox,
                              QCheckBox, QSlider, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# Import core analysis modules
from particle_tracker import ParticleTracker
from mobility_analyzer import MobilityAnalyzer
from nanopyx_processor import NanoPyxProcessor


class ProcessingThread(QThread):
    """Background thread for computationally intensive processing"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, task, *args, **kwargs):
        super().__init__()
        self.task = task
        self.args = args
        self.kwargs = kwargs
        self._is_running = True
    
    def run(self):
        try:
            if self._is_running:
                result = self.task(*self.args, **self.kwargs)
                if self._is_running:
                    self.finished.emit(result)
        except Exception as e:
            if self._is_running:
                self.error.emit(str(e))
    
    def stop(self):
        """Stop the thread gracefully"""
        self._is_running = False
        self.wait()


class ImageViewer(pg.ImageView):
    """Enhanced image viewer with particle overlay capabilities"""
    
    def __init__(self):
        super().__init__()
        self.particles = None
        self.tracks = None
        
        # Disable auto-range and auto-levels to prevent jumping
        self.view.setAspectLocked(True)
        self.ui.histogram.setLevels(0, 1000)  # Will be updated when image is set
        
        # Create scatter plot for particles
        self.scatter_plot = pg.ScatterPlotItem()
        self.scatter_plot.setZValue(10)  # Draw on top
        self.view.addItem(self.scatter_plot)
        
        # Create plot items for tracks
        self.track_lines = []
        
    def setImage(self, img, *args, **kwargs):
        """Override setImage to control auto-ranging"""
        # Set autoRange and autoLevels only for first frame
        if not hasattr(self, '_first_image_set'):
            kwargs['autoRange'] = True
            kwargs['autoLevels'] = True
            kwargs['autoHistogramRange'] = True
            self._first_image_set = True
        else:
            kwargs['autoRange'] = False
            kwargs['autoLevels'] = False
            kwargs['autoHistogramRange'] = False
        
        super().setImage(img, *args, **kwargs)
        
    def set_particles(self, particles, frame_idx=None):
        """Display detected particles as overlay"""
        self.particles = particles
        
        if particles is None or len(particles) == 0:
            self.scatter_plot.setData(pos=np.array([]))
            return
        
        # If no frame specified, use current frame
        if frame_idx is None:
            frame_idx = self.currentIndex
        
        # Filter particles for this frame
        frame_particles = particles[particles['frame'] == frame_idx]
        
        if len(frame_particles) > 0:
            # Trackpy uses: 'y' = row (vertical), 'x' = column (horizontal)
            # pyqtgraph scatter expects: (x, y) positions
            # If coordinates appear swapped, toggle the order below:
            
            # Try this first:
            pos = np.column_stack((frame_particles['x'].values, 
                                  frame_particles['y'].values))
            
            # If reversed, uncomment this instead:
            # pos = np.column_stack((frame_particles['y'].values, 
            #                       frame_particles['x'].values))
            
            # Create red circles for detected particles
            self.scatter_plot.setData(
                pos=pos,
                size=10,
                pen=pg.mkPen('r', width=2),
                brush=None,
                symbol='o'
            )
        else:
            self.scatter_plot.setData(pos=np.array([]))
    
    def set_tracks(self, tracks, frame_idx=None, max_tracks=100):
        """Display tracks as overlay"""
        self.tracks = tracks
        
        # Clear old track lines
        for line in self.track_lines:
            try:
                self.view.removeItem(line)
            except:
                pass
        self.track_lines = []
        
        if tracks is None or len(tracks) == 0:
            return
        
        # If no frame specified, use current frame
        if frame_idx is None:
            frame_idx = self.currentIndex
        
        # Show tracks up to current frame
        current_tracks = tracks[tracks['frame'] <= frame_idx]
        
        if len(current_tracks) == 0:
            return
        
        # Limit number of tracks to display (performance)
        unique_particles = current_tracks['particle'].unique()
        if len(unique_particles) > max_tracks:
            # Sample random subset
            unique_particles = random.sample(list(unique_particles), max_tracks)
        
        # Draw each track
        n_colors = min(len(unique_particles), 20)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_colors))
        
        tracks_drawn = 0
        for i, particle_id in enumerate(unique_particles):
            particle_track = current_tracks[current_tracks['particle'] == particle_id]
            
            if len(particle_track) > 1:
                # Sort by frame
                particle_track = particle_track.sort_values('frame')
                
                # Get coordinates - use same convention as particles
                # If particles are correct but tracks are wrong, swap x and y below
                x = particle_track['x'].values
                y = particle_track['y'].values
                
                # If coordinates appear swapped, uncomment these instead:
                # x = particle_track['y'].values
                # y = particle_track['x'].values
                
                # Create path
                color_idx = i % n_colors
                path = pg.PlotCurveItem(
                    x=x,
                    y=y,
                    pen=pg.mkPen(
                        color=[int(c*255) for c in colors[color_idx][:3]], 
                        width=2
                    ),
                    antialias=True
                )
                path.setZValue(5)  # Below particles but above image
                self.view.addItem(path)
                self.track_lines.append(path)
                tracks_drawn += 1
        
        print(f"Drew {tracks_drawn} tracks from {len(unique_particles)} particles")  # Debug
    
    def update_overlays(self):
        """Update overlays when frame changes"""
        current_frame = self.currentIndex
        
        # Check if we're in the main window and respect display settings
        # This is a bit hacky but works - we look for the parent window
        try:
            main_window = self.window()
            if hasattr(main_window, 'check_show_particles'):
                if main_window.check_show_particles.isChecked() and self.particles is not None:
                    self.set_particles(self.particles, current_frame)
                else:
                    self.scatter_plot.setData(pos=np.array([]))
                
                if main_window.check_show_tracks.isChecked() and self.tracks is not None:
                    max_tracks = main_window.spin_max_tracks.value()
                    self.set_tracks(self.tracks, current_frame, max_tracks=max_tracks)
                else:
                    for line in self.track_lines:
                        self.view.removeItem(line)
                    self.track_lines = []
            else:
                # Fallback if not in main window
                if self.particles is not None:
                    self.set_particles(self.particles, current_frame)
                if self.tracks is not None:
                    self.set_tracks(self.tracks, current_frame)
        except:
            # Fallback
            if self.particles is not None:
                self.set_particles(self.particles, current_frame)
            if self.tracks is not None:
                self.set_tracks(self.tracks, current_frame)


class MSDPlot(FigureCanvasQTAgg):
    """Mean Square Displacement plot widget"""
    
    def __init__(self):
        fig = Figure(figsize=(6, 4))
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.axes.set_xlabel('Time lag (s)')
        self.axes.set_ylabel('MSD (μm²)')
        self.axes.set_title('Mean Square Displacement')
        
    def plot_msd(self, msd_data, ensemble=True):
        """Plot MSD curves"""
        self.axes.clear()
        
        if ensemble:
            # Handle both Series and DataFrame
            if hasattr(msd_data, 'values'):
                # It's a Series or DataFrame
                if isinstance(msd_data, pd.Series):
                    msd_values = msd_data.values
                elif 'msd' in msd_data.columns:
                    msd_values = msd_data['msd'].values
                else:
                    # Take first column or the values directly
                    msd_values = msd_data.iloc[:, 0].values if len(msd_data.columns) > 0 else msd_data.values
                
                self.axes.plot(msd_data.index, msd_values, 'b-', linewidth=2, label='Ensemble MSD')
                
                # Plot theoretical Brownian motion if D is known
                if hasattr(msd_data, 'attrs') and 'D' in msd_data.attrs:
                    D = msd_data.attrs['D']
                    theory = 4 * D * msd_data.index
                    self.axes.plot(msd_data.index, theory, 'r--', label=f'Brownian (D={D:.3e})')
        else:
            # Plot individual trajectory MSDs
            if isinstance(msd_data, pd.DataFrame):
                for col in msd_data.columns:
                    self.axes.plot(msd_data.index, msd_data[col], alpha=0.3)
        
        self.axes.set_xlabel('Time lag (s)')
        self.axes.set_ylabel('MSD (μm²)')
        self.axes.set_title('Mean Square Displacement')
        self.axes.legend()
        self.axes.grid(True, alpha=0.3)
        self.draw()


class DiffusionHistogram(FigureCanvasQTAgg):
    """Histogram of diffusion coefficients"""
    
    def __init__(self):
        fig = Figure(figsize=(6, 4))
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        
    def plot_histogram(self, diffusion_data, mobility_classes=None):
        """Plot histogram of diffusion coefficients colored by mobility type"""
        self.axes.clear()
        
        if mobility_classes is not None:
            # Color by mobility class
            classes = ['brownian', 'confined', 'directed', 'subdiffusive', 'superdiffusive']
            colors = ['blue', 'green', 'red', 'orange', 'purple']
            
            for cls, color in zip(classes, colors):
                mask = mobility_classes == cls
                if mask.sum() > 0:
                    self.axes.hist(diffusion_data[mask], bins=30, alpha=0.6, 
                                 label=cls.capitalize(), color=color)
        else:
            self.axes.hist(diffusion_data, bins=30, alpha=0.7, color='steelblue')
        
        self.axes.set_xlabel('Diffusion Coefficient (μm²/s)')
        self.axes.set_ylabel('Count')
        self.axes.set_title('Distribution of Diffusion Coefficients')
        self.axes.legend()
        self.axes.grid(True, alpha=0.3)
        self.draw()


class ParticleTrackingGUI(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Particle Tracking with NanoPyx")
        self.setGeometry(100, 100, 1600, 900)
        
        # Initialize processors
        self.nanopyx_processor = NanoPyxProcessor()
        self.particle_tracker = ParticleTracker()
        self.mobility_analyzer = MobilityAnalyzer()
        
        # Data storage
        self.image_stack = None
        self.processed_stack = None
        self.particles = None
        self.tracks = None
        self.mobility_results = None
        
        # Thread management
        self.active_threads = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)
        
        # Main content area with tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_image_tab(), "Image Processing")
        self.tabs.addTab(self.create_tracking_tab(), "Particle Tracking")
        self.tabs.addTab(self.create_analysis_tab(), "Mobility Analysis")
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def create_control_panel(self):
        """Create top control panel with main actions"""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        
        # File operations
        self.btn_load = QPushButton("Load Image Stack")
        self.btn_load.clicked.connect(self.load_image_stack)
        layout.addWidget(self.btn_load)
        
        # Processing pipeline
        self.btn_process = QPushButton("Process with NanoPyx")
        self.btn_process.clicked.connect(self.process_with_nanopyx)
        self.btn_process.setEnabled(False)
        layout.addWidget(self.btn_process)
        
        self.btn_track = QPushButton("Detect & Track Particles")
        self.btn_track.clicked.connect(self.track_particles)
        self.btn_track.setEnabled(False)
        layout.addWidget(self.btn_track)
        
        self.btn_analyze = QPushButton("Analyze Mobility")
        self.btn_analyze.clicked.connect(self.analyze_mobility)
        self.btn_analyze.setEnabled(False)
        layout.addWidget(self.btn_analyze)
        
        layout.addStretch()
        
        # Export
        self.btn_export = QPushButton("Export Results")
        self.btn_export.clicked.connect(self.export_results)
        self.btn_export.setEnabled(False)
        layout.addWidget(self.btn_export)
        
        return panel
    
    def create_image_tab(self):
        """Create image processing tab"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left: Image viewer
        self.image_viewer = ImageViewer()
        layout.addWidget(self.image_viewer, stretch=3)
        
        # Right: Processing parameters
        params_panel = QGroupBox("NanoPyx Parameters")
        params_layout = QFormLayout()
        
        self.check_drift_correction = QCheckBox("Apply Drift Correction")
        self.check_drift_correction.setChecked(True)
        params_layout.addRow(self.check_drift_correction)
        
        self.check_esrrf = QCheckBox("Apply eSRRF Enhancement")
        params_layout.addRow(self.check_esrrf)
        
        self.spin_esrrf_magnification = QSpinBox()
        self.spin_esrrf_magnification.setRange(1, 10)
        self.spin_esrrf_magnification.setValue(5)
        params_layout.addRow("eSRRF Magnification:", self.spin_esrrf_magnification)
        
        self.spin_esrrf_radius = QDoubleSpinBox()
        self.spin_esrrf_radius.setRange(0.5, 5.0)
        self.spin_esrrf_radius.setValue(1.5)
        self.spin_esrrf_radius.setSingleStep(0.1)
        params_layout.addRow("eSRRF Radius:", self.spin_esrrf_radius)
        
        params_panel.setLayout(params_layout)
        params_panel.setMaximumWidth(300)
        layout.addWidget(params_panel, stretch=1)
        
        return tab
    
    def create_tracking_tab(self):
        """Create particle tracking tab"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left: Image with overlays
        self.tracking_viewer = ImageViewer()
        # Connect frame changes to overlay updates
        self.tracking_viewer.sigTimeChanged.connect(
            lambda: self.tracking_viewer.update_overlays()
        )
        layout.addWidget(self.tracking_viewer, stretch=3)
        
        # Right: Tracking parameters
        params_panel = QGroupBox("Tracking Parameters")
        params_layout = QFormLayout()
        
        self.spin_diameter = QSpinBox()
        self.spin_diameter.setRange(3, 51)
        self.spin_diameter.setValue(11)
        self.spin_diameter.setSingleStep(2)  # Keep odd
        params_layout.addRow("Particle Diameter (px):", self.spin_diameter)
        
        self.spin_minmass = QSpinBox()
        self.spin_minmass.setRange(0, 10000)
        self.spin_minmass.setValue(100)
        params_layout.addRow("Min Mass:", self.spin_minmass)
        
        self.spin_search_range = QDoubleSpinBox()
        self.spin_search_range.setRange(1, 50)
        self.spin_search_range.setValue(5.0)
        params_layout.addRow("Search Range (px):", self.spin_search_range)
        
        self.spin_memory = QSpinBox()
        self.spin_memory.setRange(0, 10)
        self.spin_memory.setValue(3)
        params_layout.addRow("Memory (frames):", self.spin_memory)
        
        self.spin_min_track_length = QSpinBox()
        self.spin_min_track_length.setRange(2, 1000)
        self.spin_min_track_length.setValue(10)
        params_layout.addRow("Min Track Length:", self.spin_min_track_length)
        
        # Overlay controls
        params_layout.addRow(QLabel(""))  # Spacer
        params_layout.addRow(QLabel("Display Options:"))
        
        self.check_show_particles = QCheckBox("Show Particles")
        self.check_show_particles.setChecked(True)
        self.check_show_particles.stateChanged.connect(self.update_tracking_display)
        params_layout.addRow(self.check_show_particles)
        
        self.check_show_tracks = QCheckBox("Show Tracks")
        self.check_show_tracks.setChecked(True)
        self.check_show_tracks.stateChanged.connect(self.update_tracking_display)
        params_layout.addRow(self.check_show_tracks)
        
        self.spin_max_tracks = QSpinBox()
        self.spin_max_tracks.setRange(10, 1000)
        self.spin_max_tracks.setValue(100)
        self.spin_max_tracks.setSingleStep(10)
        self.spin_max_tracks.valueChanged.connect(self.update_tracking_display)
        params_layout.addRow("Max Tracks to Display:", self.spin_max_tracks)
        
        # Refresh button
        self.btn_refresh_display = QPushButton("Refresh Display")
        self.btn_refresh_display.clicked.connect(self.update_tracking_display)
        params_layout.addRow(self.btn_refresh_display)
        
        params_panel.setLayout(params_layout)
        params_panel.setMaximumWidth(300)
        layout.addWidget(params_panel, stretch=1)
        
        return tab
    
    def create_analysis_tab(self):
        """Create mobility analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Top: Parameters
        params_panel = QGroupBox("Analysis Parameters")
        params_layout = QFormLayout()
        
        self.spin_pixel_size = QDoubleSpinBox()
        self.spin_pixel_size.setRange(0.001, 1.0)
        self.spin_pixel_size.setValue(0.065)  # 65 nm typical for sCMOS
        self.spin_pixel_size.setDecimals(3)
        params_layout.addRow("Pixel Size (μm):", self.spin_pixel_size)
        
        self.spin_frame_rate = QDoubleSpinBox()
        self.spin_frame_rate.setRange(0.1, 1000)
        self.spin_frame_rate.setValue(10.0)  # 10 Hz
        params_layout.addRow("Frame Rate (Hz):", self.spin_frame_rate)
        
        self.spin_max_lagtime = QSpinBox()
        self.spin_max_lagtime.setRange(1, 100)
        self.spin_max_lagtime.setValue(20)
        params_layout.addRow("Max Lag Time (frames):", self.spin_max_lagtime)
        
        params_panel.setLayout(params_layout)
        layout.addWidget(params_panel)
        
        # Middle: Plots
        plot_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.msd_plot = MSDPlot()
        plot_splitter.addWidget(self.msd_plot)
        
        self.diff_histogram = DiffusionHistogram()
        plot_splitter.addWidget(self.diff_histogram)
        
        layout.addWidget(plot_splitter, stretch=3)
        
        # Bottom: Results summary
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        layout.addWidget(self.results_text, stretch=1)
        
        return tab
    
    def load_image_stack(self):
        """Load microscopy image stack"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Image Stack", "",
            "Image Files (*.tif *.tiff *.nd2);;All Files (*)"
        )
        
        if file_path:
            try:
                # Load image stack (implement based on file format)
                from skimage import io
                self.image_stack = io.imread(file_path)
                
                if self.image_stack.ndim == 2:
                    self.image_stack = self.image_stack[np.newaxis, ...]
                
                # Display first frame
                self.image_viewer.setImage(self.image_stack)
                self.tracking_viewer.setImage(self.image_stack)
                
                self.statusBar().showMessage(
                    f"Loaded {self.image_stack.shape[0]} frames, "
                    f"size: {self.image_stack.shape[1]}x{self.image_stack.shape[2]}"
                )
                
                self.btn_process.setEnabled(True)
                
            except Exception as e:
                self.statusBar().showMessage(f"Error loading image: {str(e)}")
    
    def process_with_nanopyx(self):
        """Process images using NanoPyx"""
        if self.image_stack is None:
            return
        
        def process():
            # Apply drift correction
            if self.check_drift_correction.isChecked():
                stack = self.nanopyx_processor.drift_correction(self.image_stack)
            else:
                stack = self.image_stack
            
            # Apply eSRRF
            if self.check_esrrf.isChecked():
                stack = self.nanopyx_processor.esrrf_enhancement(
                    stack,
                    magnification=self.spin_esrrf_magnification.value(),
                    radius=self.spin_esrrf_radius.value()
                )
            
            return stack
        
        self.progress_bar.setValue(0)
        thread = ProcessingThread(process)
        thread.finished.connect(self.on_processing_finished)
        thread.error.connect(lambda e: self.statusBar().showMessage(f"Error: {e}"))
        self.active_threads.append(thread)
        thread.finished.connect(lambda: self.cleanup_thread(thread))
        thread.error.connect(lambda: self.cleanup_thread(thread))
        thread.start()
        
    def on_processing_finished(self, processed_stack):
        """Handle completed image processing"""
        self.processed_stack = processed_stack
        self.tracking_viewer.setImage(processed_stack)
        self.progress_bar.setValue(100)
        self.statusBar().showMessage("Processing complete")
        self.btn_track.setEnabled(True)
    
    def track_particles(self):
        """Detect and track particles"""
        stack = self.processed_stack if self.processed_stack is not None else self.image_stack
        
        if stack is None:
            return
        
        def track():
            # Detect particles
            particles = self.particle_tracker.detect_particles(
                stack,
                diameter=self.spin_diameter.value(),
                minmass=self.spin_minmass.value()
            )
            
            # Link into tracks
            tracks = self.particle_tracker.link_particles(
                particles,
                search_range=self.spin_search_range.value(),
                memory=self.spin_memory.value()
            )
            
            # Filter short tracks
            tracks = self.particle_tracker.filter_tracks(
                tracks,
                min_length=self.spin_min_track_length.value()
            )
            
            return particles, tracks
        
        self.progress_bar.setValue(0)
        thread = ProcessingThread(track)
        thread.finished.connect(self.on_tracking_finished)
        thread.error.connect(lambda e: self.statusBar().showMessage(f"Error: {e}"))
        self.active_threads.append(thread)
        thread.finished.connect(lambda: self.cleanup_thread(thread))
        thread.error.connect(lambda: self.cleanup_thread(thread))
        thread.start()
    
    def on_tracking_finished(self, result):
        """Handle completed particle tracking"""
        self.particles, self.tracks = result
        
        # Get current frame from viewer
        current_frame = self.tracking_viewer.currentIndex
        
        # Update display with overlays
        self.tracking_viewer.set_particles(self.particles, current_frame)
        self.tracking_viewer.set_tracks(self.tracks, current_frame)
        
        n_particles = len(self.particles)
        n_tracks = self.tracks['particle'].nunique()
        self.statusBar().showMessage(
            f"Detected {n_particles} particles, linked into {n_tracks} tracks"
        )
        
        self.progress_bar.setValue(100)
        self.btn_analyze.setEnabled(True)
    
    def update_tracking_display(self):
        """Update particle and track display based on checkbox states"""
        if not hasattr(self, 'particles'):
            return
        
        if self.particles is None and self.tracks is None:
            return
        
        current_frame = self.tracking_viewer.currentIndex
        print(f"Updating display for frame {current_frame}")  # Debug
        
        # Show/hide particles
        if self.check_show_particles.isChecked() and self.particles is not None:
            self.tracking_viewer.set_particles(self.particles, current_frame)
            print(f"  Showing particles")  # Debug
        else:
            self.tracking_viewer.scatter_plot.setData(pos=np.array([]))
            print(f"  Hiding particles")  # Debug
        
        # Show/hide tracks
        if self.check_show_tracks.isChecked() and self.tracks is not None:
            max_tracks = self.spin_max_tracks.value()
            self.tracking_viewer.set_tracks(self.tracks, current_frame, max_tracks=max_tracks)
            print(f"  Showing tracks (max={max_tracks})")  # Debug
        else:
            # Clear tracks
            for line in self.tracking_viewer.track_lines:
                try:
                    self.tracking_viewer.view.removeItem(line)
                except:
                    pass
            self.tracking_viewer.track_lines = []
            print(f"  Hiding tracks")  # Debug
    
    def analyze_mobility(self):
        """Analyze particle mobility"""
        if self.tracks is None:
            return
        
        def analyze():
            # Calculate MSD
            msd_data = self.mobility_analyzer.calculate_msd(
                self.tracks,
                pixel_size=self.spin_pixel_size.value(),
                fps=self.spin_frame_rate.value(),
                max_lagtime=self.spin_max_lagtime.value()
            )
            
            # Classify mobility
            mobility_classes = self.mobility_analyzer.classify_mobility(
                self.tracks,
                pixel_size=self.spin_pixel_size.value(),
                fps=self.spin_frame_rate.value()
            )
            
            # Calculate diffusion coefficients
            diffusion_coeffs = self.mobility_analyzer.calculate_diffusion_coefficients(
                self.tracks,
                pixel_size=self.spin_pixel_size.value(),
                fps=self.spin_frame_rate.value()
            )
            
            return msd_data, mobility_classes, diffusion_coeffs
        
        self.progress_bar.setValue(0)
        thread = ProcessingThread(analyze)
        thread.finished.connect(self.on_analysis_finished)
        thread.error.connect(lambda e: self.statusBar().showMessage(f"Error: {e}"))
        self.active_threads.append(thread)
        thread.finished.connect(lambda: self.cleanup_thread(thread))
        thread.error.connect(lambda: self.cleanup_thread(thread))
        thread.start()
    
    def on_analysis_finished(self, result):
        """Handle completed mobility analysis"""
        msd_data, mobility_classes, diffusion_coeffs = result
        self.mobility_results = {
            'msd': msd_data,
            'classes': mobility_classes,
            'diffusion': diffusion_coeffs
        }
        
        # Update plots
        self.msd_plot.plot_msd(msd_data)
        self.diff_histogram.plot_histogram(diffusion_coeffs, mobility_classes)
        
        # Generate summary
        summary = self.generate_results_summary(mobility_classes, diffusion_coeffs)
        self.results_text.setText(summary)
        
        self.progress_bar.setValue(100)
        self.btn_export.setEnabled(True)
        self.statusBar().showMessage("Analysis complete")
    
    def generate_results_summary(self, mobility_classes, diffusion_coeffs):
        """Generate text summary of results"""
        summary = "=== Mobility Analysis Results ===\n\n"
        
        # Count each mobility type
        from collections import Counter
        class_counts = Counter(mobility_classes)
        
        summary += f"Total tracks analyzed: {len(mobility_classes)}\n\n"
        summary += "Mobility Classification:\n"
        for cls, count in class_counts.items():
            pct = 100 * count / len(mobility_classes)
            summary += f"  {cls.capitalize()}: {count} ({pct:.1f}%)\n"
        
        summary += f"\nDiffusion Coefficients:\n"
        summary += f"  Mean: {np.mean(diffusion_coeffs):.3e} μm²/s\n"
        summary += f"  Median: {np.median(diffusion_coeffs):.3e} μm²/s\n"
        summary += f"  Std: {np.std(diffusion_coeffs):.3e} μm²/s\n"
        summary += f"  Range: [{np.min(diffusion_coeffs):.3e}, {np.max(diffusion_coeffs):.3e}] μm²/s\n"
        
        return summary
    
    def cleanup_thread(self, thread):
        """Remove thread from active list"""
        if thread in self.active_threads:
            self.active_threads.remove(thread)
    
    def closeEvent(self, event):
        """Clean up threads before closing"""
        # Stop all active threads
        for thread in self.active_threads:
            if thread.isRunning():
                thread.stop()
        
        # Wait for all threads to finish
        for thread in self.active_threads:
            thread.wait(1000)  # Wait max 1 second per thread
        
        event.accept()
    
    def export_results(self):
        """Export analysis results"""
        if self.tracks is None or self.mobility_results is None:
            return
        
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if output_dir:
            import pandas as pd
            
            # Get mobility classes for each particle
            mobility_classes = self.mobility_results['classes']
            
            # Export tracks with mobility classification
            tracks_with_mobility = self.tracks.copy()
            # Map particle IDs to mobility classes
            particle_to_class = {i: cls for i, cls in enumerate(mobility_classes)}
            tracks_with_mobility['mobility_class'] = tracks_with_mobility['particle'].map(particle_to_class)
            tracks_with_mobility.to_csv(f"{output_dir}/tracks_with_mobility.csv", index=False)
            
            # Export diffusion coefficients
            diff_df = pd.DataFrame({
                'particle': range(len(self.mobility_results['diffusion'])),
                'diffusion_coefficient': self.mobility_results['diffusion'],
                'mobility_class': self.mobility_results['classes']
            })
            diff_df.to_csv(f"{output_dir}/diffusion_coefficients.csv", index=False)
            
            # Export MSD data
            self.mobility_results['msd'].to_csv(f"{output_dir}/msd_data.csv")
            
            self.statusBar().showMessage(f"Results exported to {output_dir}")


def main():
    app = QApplication(sys.argv)
    window = ParticleTrackingGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
