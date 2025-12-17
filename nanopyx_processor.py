"""
NanoPyx Image Processor
========================

Integrates NanoPyx's high-performance image processing capabilities:
- Drift correction using cross-correlation
- eSRRF (enhanced SRRF) super-resolution
- Image registration and channel alignment
- Quality control metrics (FRC, decorrelation)
"""

import numpy as np
from typing import Tuple, Optional


class NanoPyxProcessor:
    """
    Wrapper for NanoPyx image processing methods with adaptive performance optimization
    """
    
    def __init__(self):
        """Initialize NanoPyx processor with Liquid Engine"""
        try:
            import nanopyx
            from nanopyx.core.transform import ShiftAndMagnify
            from nanopyx.core.transform.sr_temporal_correlations import calculate_SRRF
            self.nanopyx_available = True
            self.shift_magnify = ShiftAndMagnify()
            print("✓ NanoPyx loaded successfully")
        except ImportError as e:
            print(f"Warning: NanoPyx not installed. Using fallback methods.")
            print(f"  Error: {e}")
            self.nanopyx_available = False
        except Exception as e:
            print(f"Warning: NanoPyx installed but failed to initialize: {e}")
            print("  This is normal on Mac without CUDA. Using fallback methods.")
            self.nanopyx_available = False
    
    def drift_correction(self, image_stack: np.ndarray, 
                        reference_frame: Optional[int] = None) -> np.ndarray:
        """
        Correct drift in image stack using cross-correlation
        
        Parameters
        ----------
        image_stack : np.ndarray
            3D array (frames, height, width)
        reference_frame : int, optional
            Index of reference frame. If None, uses middle frame
            
        Returns
        -------
        corrected_stack : np.ndarray
            Drift-corrected image stack
        """
        if self.nanopyx_available:
            return self._nanopyx_drift_correction(image_stack, reference_frame)
        else:
            return self._fallback_drift_correction(image_stack, reference_frame)
    
    def _nanopyx_drift_correction(self, image_stack: np.ndarray,
                                  reference_frame: Optional[int] = None) -> np.ndarray:
        """Use NanoPyx's optimized drift correction"""
        try:
            from nanopyx.methods import DriftCorrection
            
            if reference_frame is None:
                reference_frame = image_stack.shape[0] // 2
            
            # NanoPyx drift correction with Liquid Engine optimization
            dc = DriftCorrection()
            corrected = dc.estimate_drift_and_correct(
                image_stack,
                reference_frame=reference_frame
            )
            
            return corrected
            
        except Exception as e:
            print(f"NanoPyx drift correction failed: {e}. Using fallback.")
            return self._fallback_drift_correction(image_stack, reference_frame)
    
    def _fallback_drift_correction(self, image_stack: np.ndarray,
                                   reference_frame: Optional[int] = None) -> np.ndarray:
        """Fallback drift correction using phase correlation"""
        from scipy.ndimage import shift as nd_shift
        from skimage.registration import phase_cross_correlation
        
        if reference_frame is None:
            reference_frame = image_stack.shape[0] // 2
        
        reference = image_stack[reference_frame]
        corrected_stack = np.zeros_like(image_stack)
        
        for i, frame in enumerate(image_stack):
            # Calculate shift using phase correlation
            shift, error, phasediff = phase_cross_correlation(
                reference, frame, upsample_factor=10
            )
            
            # Apply shift
            corrected_stack[i] = nd_shift(frame, shift, mode='constant', cval=0)
        
        return corrected_stack
    
    def esrrf_enhancement(self, image_stack: np.ndarray,
                         magnification: int = 5,
                         radius: float = 1.5,
                         sensitivity: float = 2.0) -> np.ndarray:
        """
        Apply eSRRF (enhanced Super-Resolution Radial Fluctuations)
        
        Parameters
        ----------
        image_stack : np.ndarray
            Input image stack (frames, height, width)
        magnification : int
            Super-resolution magnification factor
        radius : float
            Radial gradient radius (in pixels)
        sensitivity : float
            SRRF sensitivity parameter
            
        Returns
        -------
        enhanced : np.ndarray
            Super-resolved image
        """
        if self.nanopyx_available:
            return self._nanopyx_esrrf(image_stack, magnification, radius, sensitivity)
        else:
            # Fallback: return interpolated image
            from scipy.ndimage import zoom
            print("eSRRF not available. Returning interpolated image.")
            # Average over time and interpolate
            avg_image = np.mean(image_stack, axis=0)
            return zoom(avg_image, magnification, order=3)
    
    def _nanopyx_esrrf(self, image_stack: np.ndarray,
                      magnification: int,
                      radius: float,
                      sensitivity: float) -> np.ndarray:
        """Use NanoPyx's eSRRF with Liquid Engine optimization"""
        try:
            from nanopyx.methods import eSRRF
            
            # Initialize eSRRF with Liquid Engine
            esrrf = eSRRF()
            
            # Run eSRRF - automatically uses fastest implementation
            enhanced = esrrf.calculate(
                image_stack,
                magnification=magnification,
                radius=radius,
                sensitivity=sensitivity,
                doIntensityWeighting=True
            )
            
            return enhanced
            
        except Exception as e:
            print(f"NanoPyx eSRRF failed: {e}")
            from scipy.ndimage import zoom
            avg_image = np.mean(image_stack, axis=0)
            return zoom(avg_image, magnification, order=3)
    
    def calculate_frc_resolution(self, image1: np.ndarray,
                                image2: np.ndarray,
                                pixel_size: float = 1.0) -> Tuple[float, np.ndarray]:
        """
        Calculate resolution using Fourier Ring Correlation
        
        Parameters
        ----------
        image1, image2 : np.ndarray
            Two independent reconstructions
        pixel_size : float
            Pixel size in microns
            
        Returns
        -------
        resolution : float
            Estimated resolution in microns
        frc_curve : np.ndarray
            FRC curve (frequency, correlation)
        """
        if self.nanopyx_available:
            try:
                from nanopyx.methods import FRC
                
                frc = FRC()
                resolution, frc_curve = frc.calculate(
                    image1, image2, pixel_size=pixel_size
                )
                
                return resolution, frc_curve
                
            except Exception as e:
                print(f"FRC calculation failed: {e}")
                return None, None
        else:
            print("FRC calculation requires NanoPyx")
            return None, None
    
    def calculate_decorrelation(self, image: np.ndarray,
                               pixel_size: float = 1.0) -> Tuple[float, np.ndarray]:
        """
        Calculate resolution using Image Decorrelation Analysis
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        pixel_size : float
            Pixel size in microns
            
        Returns
        -------
        resolution : float
            Estimated resolution in microns
        decorr_curve : np.ndarray
            Decorrelation curve
        """
        if self.nanopyx_available:
            try:
                from nanopyx.methods import DecorrelationAnalysis
                
                decorr = DecorrelationAnalysis()
                resolution, decorr_curve = decorr.calculate(
                    image, pixel_size=pixel_size
                )
                
                return resolution, decorr_curve
                
            except Exception as e:
                print(f"Decorrelation analysis failed: {e}")
                return None, None
        else:
            print("Decorrelation analysis requires NanoPyx")
            return None, None
    
    def channel_registration(self, channel1: np.ndarray,
                           channel2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Register two channels for multi-color imaging
        
        Parameters
        ----------
        channel1, channel2 : np.ndarray
            Images from different channels
            
        Returns
        -------
        registered_ch1, registered_ch2 : np.ndarray
            Registered channel images
        """
        if self.nanopyx_available:
            try:
                from nanopyx.methods import ChannelRegistration
                
                reg = ChannelRegistration()
                registered = reg.register(channel1, channel2)
                
                return registered[0], registered[1]
                
            except Exception as e:
                print(f"Channel registration failed: {e}")
                return channel1, channel2
        else:
            # Fallback: simple cross-correlation registration
            from scipy.ndimage import shift as nd_shift
            from skimage.registration import phase_cross_correlation
            
            shift, error, phasediff = phase_cross_correlation(channel1, channel2)
            registered_ch2 = nd_shift(channel2, shift, mode='constant', cval=0)
            
            return channel1, registered_ch2
    
    def interpolate_image(self, image: np.ndarray,
                         magnification: int = 5,
                         method: str = 'catmull-rom') -> np.ndarray:
        """
        Interpolate image with high quality
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        magnification : int
            Interpolation factor
        method : str
            Interpolation method ('catmull-rom', 'lanczos', 'bicubic')
            
        Returns
        -------
        interpolated : np.ndarray
            Interpolated image
        """
        if self.nanopyx_available:
            try:
                # Use NanoPyx's optimized interpolation
                interpolated = self.shift_magnify.run(
                    image,
                    magnification=magnification,
                    apply_shift=False
                )
                return interpolated
                
            except Exception as e:
                print(f"NanoPyx interpolation failed: {e}")
                return self._fallback_interpolation(image, magnification)
        else:
            return self._fallback_interpolation(image, magnification)
    
    def _fallback_interpolation(self, image: np.ndarray,
                               magnification: int) -> np.ndarray:
        """Fallback interpolation using scipy"""
        from scipy.ndimage import zoom
        return zoom(image, magnification, order=3)
    
    def preprocess_for_tracking(self, image_stack: np.ndarray,
                               enhance: bool = True,
                               correct_drift: bool = True) -> np.ndarray:
        """
        Full preprocessing pipeline optimized for particle tracking
        
        Parameters
        ----------
        image_stack : np.ndarray
            Raw image stack
        enhance : bool
            Whether to apply eSRRF enhancement
        correct_drift : bool
            Whether to correct drift
            
        Returns
        -------
        processed : np.ndarray
            Processed image stack ready for tracking
        """
        processed = image_stack.copy()
        
        # Drift correction
        if correct_drift:
            print("Applying drift correction...")
            processed = self.drift_correction(processed)
        
        # eSRRF enhancement
        if enhance:
            print("Applying eSRRF enhancement...")
            processed = self.esrrf_enhancement(processed)
        
        return processed
