"""
Image Processor - Handles image loading and saving with transparency support
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class ImageProcessor:
    """Handles image I/O with full alpha channel support"""
    
    # Supported formats for loading
    SUPPORTED_FORMATS = {
        '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif',
        '.webp', '.gif', '.ico', '.ppm', '.pgm', '.pbm'
    }
    
    # Formats that support transparency
    ALPHA_FORMATS = {'.png', '.tiff', '.tif', '.webp', '.ico'}
    
    @staticmethod
    def load_image(file_path: str) -> Optional[np.ndarray]:
        """
        Load an image with alpha channel preservation.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            numpy array in BGRA format, or None if loading fails
        """
        path = Path(file_path)
        
        if not path.exists():
            return None
        
        if path.suffix.lower() not in ImageProcessor.SUPPORTED_FORMATS:
            return None
        
        # Load with alpha channel if present
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        
        if image is None:
            return None
        
        # Ensure consistent BGRA format
        image = ImageProcessor.ensure_bgra(image)
        
        return image
    
    @staticmethod
    def ensure_bgra(image: np.ndarray) -> np.ndarray:
        """
        Convert image to BGRA format regardless of input format.
        
        Args:
            image: Input image in any format
            
        Returns:
            Image in BGRA format
        """
        if image is None:
            return None
        
        # Grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        # Grayscale with alpha
        elif len(image.shape) == 3 and image.shape[2] == 2:
            gray = image[:, :, 0]
            alpha = image[:, :, 1]
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            image = np.dstack((bgr, alpha))
        # BGR (no alpha)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        # Already BGRA
        elif len(image.shape) == 3 and image.shape[2] == 4:
            pass  # Already in correct format
        
        return image
    
    @staticmethod
    def save_image(image: np.ndarray, file_path: str, format_ext: str = None) -> bool:
        """
        Save an image to file.
        
        Args:
            image: Image in BGRA format
            file_path: Output file path
            format_ext: Optional format extension (e.g., 'PNG', 'JPG')
            
        Returns:
            True if successful, False otherwise
        """
        if image is None:
            return False
        
        path = Path(file_path)
        
        # Determine format from extension
        ext = format_ext.lower() if format_ext else path.suffix.lower()
        if not ext.startswith('.'):
            ext = '.' + ext
        
        # Handle formats that don't support alpha
        if ext not in ImageProcessor.ALPHA_FORMATS and image.shape[2] == 4:
            # Convert BGRA to BGR (discard alpha)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        try:
            # Set quality parameters for certain formats
            params = []
            if ext in ['.jpg', '.jpeg']:
                params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            elif ext == '.png':
                params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
            elif ext == '.webp':
                params = [cv2.IMWRITE_WEBP_QUALITY, 95]
            
            success = cv2.imwrite(str(path), image, params)
            return success
        except Exception:
            return False
    
    @staticmethod
    def get_image_dimensions(image: np.ndarray) -> Tuple[int, int]:
        """
        Get image dimensions.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (width, height)
        """
        if image is None:
            return (0, 0)
        return (image.shape[1], image.shape[0])
    
    @staticmethod
    def bgra_to_rgba(image: np.ndarray) -> np.ndarray:
        """
        Convert BGRA to RGBA for Qt display.
        
        Args:
            image: Image in BGRA format
            
        Returns:
            Image in RGBA format
        """
        if image is None:
            return None
        
        if len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    @staticmethod
    def get_file_filter() -> str:
        """Get file filter string for file dialogs"""
        return (
            "All Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp *.gif *.ico);;"
            "PNG (*.png);;"
            "JPEG (*.jpg *.jpeg);;"
            "BMP (*.bmp);;"
            "TIFF (*.tiff *.tif);;"
            "WebP (*.webp);;"
            "GIF (*.gif);;"
            "All Files (*.*)"
        )

