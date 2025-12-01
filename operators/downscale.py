"""
Downscale Operator - Resize images to smaller dimensions
"""
import cv2
import numpy as np
from typing import Callable, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QCheckBox, QComboBox, QGroupBox, QLineEdit
)
from PyQt6.QtCore import Qt

from core.operator_base import OperatorBase
from core.config_manager import ConfigManager


class DownscaleOperator(OperatorBase):
    """
    Operator for downscaling/resizing images.
    
    Features:
    - Width and height input
    - Preserve aspect ratio option
    - Multiple interpolation algorithms
    - Pixel art options: binarize alpha, crop transparent, corrode, outline, pad to size
    """
    
    # Interpolation algorithms mapping
    ALGORITHMS = {
        "Nearest": cv2.INTER_NEAREST,
        "Linear": cv2.INTER_LINEAR,
        "Cubic": cv2.INTER_CUBIC,
        "Area": cv2.INTER_AREA,
        "Lanczos": cv2.INTER_LANCZOS4
    }
    
    def __init__(self):
        self._widget: Optional[QWidget] = None
        self._width_spin: Optional[QSpinBox] = None
        self._height_spin: Optional[QSpinBox] = None
        self._preserve_ratio_check: Optional[QCheckBox] = None
        self._algorithm_combo: Optional[QComboBox] = None
        
        # Pixel art options
        self._binarize_alpha_check: Optional[QCheckBox] = None
        self._crop_transparent_check: Optional[QCheckBox] = None
        self._corrode_check: Optional[QCheckBox] = None
        self._corrode_iterations_spin: Optional[QSpinBox] = None
        self._outline_check: Optional[QCheckBox] = None
        self._outline_color_edit: Optional[QLineEdit] = None
        
        # Padding options
        self._pad_check: Optional[QCheckBox] = None
        self._pad_width_spin: Optional[QSpinBox] = None
        self._pad_height_spin: Optional[QSpinBox] = None
        
        self._source_width = 0
        self._source_height = 0
        self._aspect_ratio = 1.0
        
        self._updating = False  # Flag to prevent circular updates
        self._callback: Optional[Callable] = None
    
    @property
    def name(self) -> str:
        return "Downscale"
    
    def get_widget(self) -> QWidget:
        """Create and return the parameter widget"""
        if self._widget is not None:
            return self._widget
        
        self._widget = QWidget()
        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        
        # Size group
        size_group = QGroupBox("Output Size")
        size_layout = QVBoxLayout(size_group)
        size_layout.setSpacing(4)
        
        # Width
        width_layout = QHBoxLayout()
        width_label = QLabel("Width:")
        width_label.setFixedWidth(50)
        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 65535)
        self._width_spin.setValue(100)
        self._width_spin.setSuffix(" px")
        self._width_spin.valueChanged.connect(self._on_width_changed)
        width_layout.addWidget(width_label)
        width_layout.addWidget(self._width_spin, 1)
        size_layout.addLayout(width_layout)
        
        # Height
        height_layout = QHBoxLayout()
        height_label = QLabel("Height:")
        height_label.setFixedWidth(50)
        self._height_spin = QSpinBox()
        self._height_spin.setRange(1, 65535)
        self._height_spin.setValue(100)
        self._height_spin.setSuffix(" px")
        self._height_spin.valueChanged.connect(self._on_height_changed)
        height_layout.addWidget(height_label)
        height_layout.addWidget(self._height_spin, 1)
        size_layout.addLayout(height_layout)
        
        # Preserve ratio
        self._preserve_ratio_check = QCheckBox("Preserve aspect ratio")
        self._preserve_ratio_check.setChecked(True)
        self._preserve_ratio_check.stateChanged.connect(self._on_preserve_ratio_changed)
        size_layout.addWidget(self._preserve_ratio_check)
        
        layout.addWidget(size_group)
        
        # Algorithm group
        algo_group = QGroupBox("Algorithm")
        algo_layout = QVBoxLayout(algo_group)
        algo_layout.setSpacing(4)
        
        self._algorithm_combo = QComboBox()
        self._algorithm_combo.addItems(list(self.ALGORITHMS.keys()))
        self._algorithm_combo.setCurrentText("Linear")
        self._algorithm_combo.currentTextChanged.connect(self._on_algorithm_changed)
        algo_layout.addWidget(self._algorithm_combo)
        
        # Algorithm description
        self._algo_desc = QLabel("")
        self._algo_desc.setStyleSheet("color: #666; font-size: 10px;")
        self._algo_desc.setWordWrap(True)
        algo_layout.addWidget(self._algo_desc)
        self._update_algo_description()
        
        layout.addWidget(algo_group)
        
        # Pixel Art Options
        pixel_group = QGroupBox("Pixel Art Options")
        pixel_layout = QVBoxLayout(pixel_group)
        pixel_layout.setSpacing(4)
        
        # Crop transparent borders
        self._crop_transparent_check = QCheckBox("Crop transparent borders")
        self._crop_transparent_check.setChecked(False)
        self._crop_transparent_check.setToolTip(
            "Remove transparent/empty space around the sprite before processing."
        )
        self._crop_transparent_check.stateChanged.connect(self._on_option_changed)
        pixel_layout.addWidget(self._crop_transparent_check)
        
        # Binarize alpha
        self._binarize_alpha_check = QCheckBox("Binarize alpha (remove anti-aliasing)")
        self._binarize_alpha_check.setChecked(False)
        self._binarize_alpha_check.setToolTip(
            "Threshold alpha channel to pure 0 or 255.\n"
            "Removes anti-aliasing from transparency edges."
        )
        self._binarize_alpha_check.stateChanged.connect(self._on_option_changed)
        pixel_layout.addWidget(self._binarize_alpha_check)
        
        # Corrode (erode alpha)
        corrode_layout = QHBoxLayout()
        self._corrode_check = QCheckBox("Corrode (thin by)")
        self._corrode_check.setChecked(False)
        self._corrode_check.setToolTip(
            "Erode the alpha channel to make the sprite thinner.\n"
            "Removes pixels from the edges."
        )
        self._corrode_check.stateChanged.connect(self._on_corrode_changed)
        corrode_layout.addWidget(self._corrode_check)
        
        self._corrode_iterations_spin = QSpinBox()
        self._corrode_iterations_spin.setRange(1, 10)
        self._corrode_iterations_spin.setValue(1)
        self._corrode_iterations_spin.setSuffix(" px")
        self._corrode_iterations_spin.setEnabled(False)
        self._corrode_iterations_spin.setFixedWidth(70)
        self._corrode_iterations_spin.valueChanged.connect(self._on_option_changed)
        corrode_layout.addWidget(self._corrode_iterations_spin)
        corrode_layout.addStretch()
        pixel_layout.addLayout(corrode_layout)
        
        # Outline color
        outline_layout = QHBoxLayout()
        self._outline_check = QCheckBox("Add outline")
        self._outline_check.setChecked(False)
        self._outline_check.setToolTip(
            "Add a colored 1-pixel outline around opaque areas."
        )
        self._outline_check.stateChanged.connect(self._on_outline_changed)
        outline_layout.addWidget(self._outline_check)
        
        self._outline_color_edit = QLineEdit("#000000")
        self._outline_color_edit.setFixedWidth(80)
        self._outline_color_edit.setEnabled(False)
        self._outline_color_edit.setToolTip(
            "Outline color: hex (#RRGGBB), name (black, white), or R,G,B"
        )
        self._outline_color_edit.textChanged.connect(self._on_option_changed)
        outline_layout.addWidget(self._outline_color_edit)
        outline_layout.addStretch()
        pixel_layout.addLayout(outline_layout)
        
        layout.addWidget(pixel_group)
        
        # Output Padding
        pad_group = QGroupBox("Output Padding")
        pad_layout = QVBoxLayout(pad_group)
        pad_layout.setSpacing(4)
        
        self._pad_check = QCheckBox("Pad to fixed size")
        self._pad_check.setChecked(False)
        self._pad_check.setToolTip(
            "Center the image in a fixed-size canvas.\n"
            "Useful for sprite sheets with consistent cell sizes."
        )
        self._pad_check.stateChanged.connect(self._on_pad_changed)
        pad_layout.addWidget(self._pad_check)
        
        pad_size_layout = QHBoxLayout()
        pad_w_label = QLabel("W:")
        pad_w_label.setFixedWidth(20)
        self._pad_width_spin = QSpinBox()
        self._pad_width_spin.setRange(1, 4096)
        self._pad_width_spin.setValue(64)
        self._pad_width_spin.setSuffix(" px")
        self._pad_width_spin.setEnabled(False)
        self._pad_width_spin.valueChanged.connect(self._on_option_changed)
        
        pad_h_label = QLabel("H:")
        pad_h_label.setFixedWidth(20)
        self._pad_height_spin = QSpinBox()
        self._pad_height_spin.setRange(1, 4096)
        self._pad_height_spin.setValue(64)
        self._pad_height_spin.setSuffix(" px")
        self._pad_height_spin.setEnabled(False)
        self._pad_height_spin.valueChanged.connect(self._on_option_changed)
        
        pad_size_layout.addWidget(pad_w_label)
        pad_size_layout.addWidget(self._pad_width_spin, 1)
        pad_size_layout.addWidget(pad_h_label)
        pad_size_layout.addWidget(self._pad_height_spin, 1)
        pad_layout.addLayout(pad_size_layout)
        
        layout.addWidget(pad_group)
        
        # Spacer
        layout.addStretch()
        
        return self._widget
    
    def _update_algo_description(self):
        """Update algorithm description text"""
        descriptions = {
            "Nearest": "Fast, pixelated. Good for pixel art.",
            "Linear": "Balanced speed and quality.",
            "Cubic": "Smoother than linear, slightly slower.",
            "Area": "Best for downscaling. Reduces aliasing.",
            "Lanczos": "High quality, slowest. Best for upscaling."
        }
        algo = self._algorithm_combo.currentText()
        self._algo_desc.setText(descriptions.get(algo, ""))
    
    def _on_width_changed(self, value: int):
        """Handle width value change"""
        if self._updating:
            return
        
        if self._preserve_ratio_check.isChecked() and self._aspect_ratio > 0:
            self._updating = True
            new_height = int(value / self._aspect_ratio)
            new_height = max(1, min(65535, new_height))
            self._height_spin.setValue(new_height)
            self._updating = False
        
        self._notify_change()
    
    def _on_height_changed(self, value: int):
        """Handle height value change"""
        if self._updating:
            return
        
        if self._preserve_ratio_check.isChecked() and self._aspect_ratio > 0:
            self._updating = True
            new_width = int(value * self._aspect_ratio)
            new_width = max(1, min(65535, new_width))
            self._width_spin.setValue(new_width)
            self._updating = False
        
        self._notify_change()
    
    def _on_preserve_ratio_changed(self, state: int):
        """Handle preserve ratio checkbox change"""
        if state == Qt.CheckState.Checked.value and self._source_width > 0:
            # Recalculate aspect ratio from current source
            self._aspect_ratio = self._source_width / self._source_height
        self._notify_change()
    
    def _on_algorithm_changed(self, text: str):
        """Handle algorithm selection change"""
        self._update_algo_description()
        self._notify_change()
    
    def _on_option_changed(self, *args):
        """Handle generic option change"""
        self._notify_change()
    
    def _on_corrode_changed(self, state: int):
        """Handle corrode checkbox change"""
        enabled = state == Qt.CheckState.Checked.value
        if self._corrode_iterations_spin:
            self._corrode_iterations_spin.setEnabled(enabled)
        self._notify_change()
    
    def _on_outline_changed(self, state: int):
        """Handle outline checkbox change"""
        enabled = state == Qt.CheckState.Checked.value
        if self._outline_color_edit:
            self._outline_color_edit.setEnabled(enabled)
        self._notify_change()
    
    def _on_pad_changed(self, state: int):
        """Handle pad checkbox change"""
        enabled = state == Qt.CheckState.Checked.value
        if self._pad_width_spin:
            self._pad_width_spin.setEnabled(enabled)
        if self._pad_height_spin:
            self._pad_height_spin.setEnabled(enabled)
        self._notify_change()
    
    def _notify_change(self):
        """Notify that parameters changed"""
        if self._callback:
            self._callback()
    
    @staticmethod
    def _parse_color(color_str: str) -> tuple:
        """Parse color string to RGB tuple"""
        if color_str is None:
            return (0, 0, 0)
        color_str = color_str.strip().lower()
        if color_str == 'black':
            return (0, 0, 0)
        if color_str == 'white':
            return (255, 255, 255)
        if color_str.startswith('#') and len(color_str) == 7:
            try:
                return tuple(int(color_str[i:i+2], 16) for i in (1, 3, 5))
            except ValueError:
                return (0, 0, 0)
        if ',' in color_str:
            parts = color_str.split(',')
            if len(parts) == 3:
                try:
                    return tuple(int(x.strip()) for x in parts)
                except ValueError:
                    return (0, 0, 0)
        return (0, 0, 0)
    
    @staticmethod
    def _crop_transparent_borders(image: np.ndarray) -> np.ndarray:
        """Crop transparent borders from RGBA image"""
        if len(image.shape) != 3 or image.shape[2] != 4:
            return image
        alpha = image[:, :, 3]
        coords = cv2.findNonZero(alpha)
        if coords is None:
            return image
        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y+h, x:x+w]
    
    @staticmethod
    def _corrode_alpha(image: np.ndarray, iterations: int = 1) -> np.ndarray:
        """Erode alpha channel to make sprite thinner"""
        if len(image.shape) != 3 or image.shape[2] != 4:
            return image
        result = image.copy()
        alpha = result[:, :, 3]
        # Cross-shaped kernel for precise 1-pixel thinning
        kernel = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=np.uint8)
        eroded_alpha = cv2.erode(alpha, kernel, iterations=iterations)
        result[:, :, 3] = eroded_alpha
        return result
    
    @staticmethod
    def _add_outline(image: np.ndarray, color: tuple) -> np.ndarray:
        """Add colored outline around opaque pixels"""
        if len(image.shape) != 3 or image.shape[2] != 4:
            return image
        result = image.copy()
        alpha = result[:, :, 3]
        # Create mask of opaque pixels
        mask = (alpha > 0).astype(np.uint8)
        # Dilate to find outline area
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        # Outline is dilated minus original
        outline_mask = (dilated - mask) > 0
        # Apply outline color (BGR order for cv2)
        result[:, :, 0][outline_mask] = color[2]  # B
        result[:, :, 1][outline_mask] = color[1]  # G
        result[:, :, 2][outline_mask] = color[0]  # R
        result[:, :, 3][outline_mask] = 255  # Make outline fully opaque
        return result
    
    @staticmethod
    def _pad_to_size(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """Pad image to target size, centered"""
        h, w = image.shape[:2]
        if w >= target_width and h >= target_height:
            return image
        
        # Create new canvas
        if len(image.shape) == 3 and image.shape[2] == 4:
            new_img = np.zeros((target_height, target_width, 4), dtype=image.dtype)
        elif len(image.shape) == 3:
            new_img = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
        else:
            new_img = np.zeros((target_height, target_width), dtype=image.dtype)
        
        # Calculate centering offset
        x_offset = max(0, (target_width - w) // 2)
        y_offset = max(0, (target_height - h) // 2)
        
        # Place image on canvas
        paste_w = min(w, target_width)
        paste_h = min(h, target_height)
        new_img[y_offset:y_offset+paste_h, x_offset:x_offset+paste_w] = image[:paste_h, :paste_w]
        
        return new_img
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Process the image with downscaling and pixel art options"""
        if image is None:
            return None
        
        if self._width_spin is None or self._height_spin is None:
            return image
        
        result = image.copy()
        
        # Step 1: Crop transparent borders (before resize)
        crop_enabled = self._crop_transparent_check and self._crop_transparent_check.isChecked()
        if crop_enabled:
            result = self._crop_transparent_borders(result)
        
        # Step 2: Corrode/erode alpha (before resize)
        corrode_enabled = self._corrode_check and self._corrode_check.isChecked()
        if corrode_enabled:
            iterations = self._corrode_iterations_spin.value() if self._corrode_iterations_spin else 1
            result = self._corrode_alpha(result, iterations)
        
        # Step 3: Resize
        target_width = self._width_spin.value()
        target_height = self._height_spin.value()
        
        # If preserve ratio is enabled, recalculate target based on current image dimensions
        # This is important after crop which may change the aspect ratio
        preserve_ratio = self._preserve_ratio_check and self._preserve_ratio_check.isChecked()
        if preserve_ratio and result.shape[0] > 0 and result.shape[1] > 0:
            current_h, current_w = result.shape[:2]
            current_aspect = current_w / current_h
            
            # Calculate which dimension to use as reference
            # Use the dimension that results in a smaller image (fit within target)
            width_from_height = int(target_height * current_aspect)
            height_from_width = int(target_width / current_aspect)
            
            if width_from_height <= target_width:
                # Height is the limiting factor
                target_width = max(1, width_from_height)
            else:
                # Width is the limiting factor
                target_height = max(1, height_from_width)
        
        algo_name = self._algorithm_combo.currentText() if self._algorithm_combo else "Linear"
        interpolation = self.ALGORITHMS.get(algo_name, cv2.INTER_LINEAR)
        result = cv2.resize(result, (target_width, target_height), interpolation=interpolation)
        
        # Step 4: Binarize alpha (after resize)
        binarize_enabled = self._binarize_alpha_check and self._binarize_alpha_check.isChecked()
        if binarize_enabled and len(result.shape) == 3 and result.shape[2] == 4:
            alpha = result[:, :, 3]
            alpha = ((alpha > 128).astype(np.uint8)) * 255
            result[:, :, 3] = alpha
        
        # Step 5: Add outline (after binarize for cleaner edges)
        outline_enabled = self._outline_check and self._outline_check.isChecked()
        if outline_enabled and len(result.shape) == 3 and result.shape[2] == 4:
            color_str = self._outline_color_edit.text() if self._outline_color_edit else "#000000"
            color = self._parse_color(color_str)
            result = self._add_outline(result, color)
        
        # Step 6: Pad to fixed size (last step)
        pad_enabled = self._pad_check and self._pad_check.isChecked()
        if pad_enabled:
            pad_w = self._pad_width_spin.value() if self._pad_width_spin else 64
            pad_h = self._pad_height_spin.value() if self._pad_height_spin else 64
            result = self._pad_to_size(result, pad_w, pad_h)
        
        return result
    
    def on_parameters_changed(self, callback: Callable[[], None]) -> None:
        """Register callback for parameter changes"""
        self._callback = callback
    
    def set_source_dimensions(self, width: int, height: int) -> None:
        """Set source image dimensions"""
        self._source_width = width
        self._source_height = height
        
        if width > 0 and height > 0:
            self._aspect_ratio = width / height
            
            # Update spinboxes with source dimensions
            self._updating = True
            if self._width_spin:
                self._width_spin.setValue(width)
            if self._height_spin:
                self._height_spin.setValue(height)
            self._updating = False
    
    def save_settings(self, config: ConfigManager) -> None:
        """Save operator settings"""
        if self._algorithm_combo:
            config.save_operator_setting(self.name, "algorithm", self._algorithm_combo.currentText())
        if self._preserve_ratio_check:
            config.save_operator_setting(self.name, "preserve_ratio", self._preserve_ratio_check.isChecked())
        
        # Pixel art options
        if self._crop_transparent_check:
            config.save_operator_setting(self.name, "crop_transparent", self._crop_transparent_check.isChecked())
        if self._binarize_alpha_check:
            config.save_operator_setting(self.name, "binarize_alpha", self._binarize_alpha_check.isChecked())
        if self._corrode_check:
            config.save_operator_setting(self.name, "corrode", self._corrode_check.isChecked())
        if self._corrode_iterations_spin:
            config.save_operator_setting(self.name, "corrode_iterations", self._corrode_iterations_spin.value())
        if self._outline_check:
            config.save_operator_setting(self.name, "outline", self._outline_check.isChecked())
        if self._outline_color_edit:
            config.save_operator_setting(self.name, "outline_color", self._outline_color_edit.text())
        
        # Padding options
        if self._pad_check:
            config.save_operator_setting(self.name, "pad", self._pad_check.isChecked())
        if self._pad_width_spin:
            config.save_operator_setting(self.name, "pad_width", self._pad_width_spin.value())
        if self._pad_height_spin:
            config.save_operator_setting(self.name, "pad_height", self._pad_height_spin.value())
    
    def load_settings(self, config: ConfigManager) -> None:
        """Load operator settings"""
        # Ensure widget is created
        self.get_widget()
        
        algorithm = config.load_operator_setting(self.name, "algorithm", "Linear")
        if self._algorithm_combo:
            index = self._algorithm_combo.findText(algorithm)
            if index >= 0:
                self._algorithm_combo.setCurrentIndex(index)
        
        preserve_ratio = config.load_operator_setting(self.name, "preserve_ratio", True)
        if self._preserve_ratio_check:
            if isinstance(preserve_ratio, str):
                preserve_ratio = preserve_ratio.lower() == "true"
            self._preserve_ratio_check.setChecked(bool(preserve_ratio))
        
        # Pixel art options
        crop_transparent = config.load_operator_setting(self.name, "crop_transparent", False)
        if self._crop_transparent_check:
            if isinstance(crop_transparent, str):
                crop_transparent = crop_transparent.lower() == "true"
            self._crop_transparent_check.setChecked(bool(crop_transparent))
        
        binarize_alpha = config.load_operator_setting(self.name, "binarize_alpha", False)
        if self._binarize_alpha_check:
            if isinstance(binarize_alpha, str):
                binarize_alpha = binarize_alpha.lower() == "true"
            self._binarize_alpha_check.setChecked(bool(binarize_alpha))
        
        corrode = config.load_operator_setting(self.name, "corrode", False)
        if self._corrode_check:
            if isinstance(corrode, str):
                corrode = corrode.lower() == "true"
            self._corrode_check.setChecked(bool(corrode))
        
        corrode_iterations = config.load_operator_setting(self.name, "corrode_iterations", 1)
        if self._corrode_iterations_spin:
            if isinstance(corrode_iterations, str):
                corrode_iterations = int(corrode_iterations)
            self._corrode_iterations_spin.setValue(int(corrode_iterations))
        
        outline = config.load_operator_setting(self.name, "outline", False)
        if self._outline_check:
            if isinstance(outline, str):
                outline = outline.lower() == "true"
            self._outline_check.setChecked(bool(outline))
        
        outline_color = config.load_operator_setting(self.name, "outline_color", "#000000")
        if self._outline_color_edit:
            self._outline_color_edit.setText(str(outline_color))
        
        # Padding options
        pad = config.load_operator_setting(self.name, "pad", False)
        if self._pad_check:
            if isinstance(pad, str):
                pad = pad.lower() == "true"
            self._pad_check.setChecked(bool(pad))
        
        pad_width = config.load_operator_setting(self.name, "pad_width", 64)
        if self._pad_width_spin:
            if isinstance(pad_width, str):
                pad_width = int(pad_width)
            self._pad_width_spin.setValue(int(pad_width))
        
        pad_height = config.load_operator_setting(self.name, "pad_height", 64)
        if self._pad_height_spin:
            if isinstance(pad_height, str):
                pad_height = int(pad_height)
            self._pad_height_spin.setValue(int(pad_height))
    
    def reset_to_defaults(self) -> None:
        """Reset to default values"""
        if self._algorithm_combo:
            self._algorithm_combo.setCurrentText("Linear")
        if self._preserve_ratio_check:
            self._preserve_ratio_check.setChecked(True)
        
        # Pixel art options
        if self._crop_transparent_check:
            self._crop_transparent_check.setChecked(False)
        if self._binarize_alpha_check:
            self._binarize_alpha_check.setChecked(False)
        if self._corrode_check:
            self._corrode_check.setChecked(False)
        if self._corrode_iterations_spin:
            self._corrode_iterations_spin.setValue(1)
        if self._outline_check:
            self._outline_check.setChecked(False)
        if self._outline_color_edit:
            self._outline_color_edit.setText("#000000")
        
        # Padding options
        if self._pad_check:
            self._pad_check.setChecked(False)
        if self._pad_width_spin:
            self._pad_width_spin.setValue(64)
        if self._pad_height_spin:
            self._pad_height_spin.setValue(64)

