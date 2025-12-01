"""
Color Quantize Operator - Reduce color palette using LAB color space
"""
import cv2
import numpy as np
from typing import Callable, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QCheckBox, QGroupBox, QSlider, QComboBox
)
from PyQt6.QtCore import Qt

from core.operator_base import OperatorBase
from core.config_manager import ConfigManager


class ColorQuantizeOperator(OperatorBase):
    """
    Operator for reducing color palette using K-Means clustering in LAB color space.
    
    Features:
    - K-Means clustering in perceptually uniform LAB space
    - Configurable number of colors
    - Adjustable L/a/b weights
    - Separate handling for achromatic colors
    - Preserve black/white option
    - Optional color snapping to clean values
    """
    
    def __init__(self):
        self._widget: Optional[QWidget] = None
        
        # Main parameters
        self._num_colors_spin: Optional[QSpinBox] = None
        self._iterations_spin: Optional[QSpinBox] = None
        
        # Weight sliders
        self._l_weight_slider: Optional[QSlider] = None
        self._a_weight_slider: Optional[QSlider] = None
        self._b_weight_slider: Optional[QSlider] = None
        self._l_weight_label: Optional[QLabel] = None
        self._a_weight_label: Optional[QLabel] = None
        self._b_weight_label: Optional[QLabel] = None
        
        # Options
        self._achromatic_threshold_spin: Optional[QSpinBox] = None
        self._preserve_bw_check: Optional[QCheckBox] = None
        self._snap_colors_check: Optional[QCheckBox] = None
        self._snap_l_spin: Optional[QSpinBox] = None
        self._snap_ab_spin: Optional[QSpinBox] = None
        
        self._callback: Optional[Callable] = None
    
    @property
    def name(self) -> str:
        return "Color Quantize"
    
    def get_widget(self) -> QWidget:
        """Create and return the parameter widget"""
        if self._widget is not None:
            return self._widget
        
        self._widget = QWidget()
        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        
        # Main Settings group
        main_group = QGroupBox("Palette Settings")
        main_layout = QVBoxLayout(main_group)
        main_layout.setSpacing(4)
        
        # Number of colors
        colors_layout = QHBoxLayout()
        colors_label = QLabel("Colors:")
        colors_label.setFixedWidth(70)
        self._num_colors_spin = QSpinBox()
        self._num_colors_spin.setRange(2, 256)
        self._num_colors_spin.setValue(16)
        self._num_colors_spin.setToolTip(
            "Target number of colors in the output palette.\n"
            "Common values: 8, 16, 32, 64 for pixel art."
        )
        self._num_colors_spin.valueChanged.connect(self._on_option_changed)
        colors_layout.addWidget(colors_label)
        colors_layout.addWidget(self._num_colors_spin, 1)
        main_layout.addLayout(colors_layout)
        
        # K-Means iterations
        iter_layout = QHBoxLayout()
        iter_label = QLabel("Iterations:")
        iter_label.setFixedWidth(70)
        self._iterations_spin = QSpinBox()
        self._iterations_spin.setRange(1, 50)
        self._iterations_spin.setValue(10)
        self._iterations_spin.setToolTip(
            "Number of K-Means iterations.\n"
            "Higher = better quality but slower.\n"
            "10 is usually sufficient."
        )
        self._iterations_spin.valueChanged.connect(self._on_option_changed)
        iter_layout.addWidget(iter_label)
        iter_layout.addWidget(self._iterations_spin, 1)
        main_layout.addLayout(iter_layout)
        
        layout.addWidget(main_group)
        
        # LAB Weights group
        weights_group = QGroupBox("LAB Channel Weights")
        weights_layout = QVBoxLayout(weights_group)
        weights_layout.setSpacing(4)
        
        weights_desc = QLabel("Adjust importance of each LAB channel")
        weights_desc.setStyleSheet("color: #666; font-size: 10px;")
        weights_layout.addWidget(weights_desc)
        
        # L weight (Lightness)
        l_layout = QHBoxLayout()
        l_name = QLabel("L (Light):")
        l_name.setFixedWidth(70)
        self._l_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self._l_weight_slider.setRange(0, 100)
        self._l_weight_slider.setValue(40)
        self._l_weight_slider.setToolTip("Lightness weight (0-100%)")
        self._l_weight_slider.valueChanged.connect(self._on_l_weight_changed)
        self._l_weight_label = QLabel("40%")
        self._l_weight_label.setFixedWidth(35)
        l_layout.addWidget(l_name)
        l_layout.addWidget(self._l_weight_slider, 1)
        l_layout.addWidget(self._l_weight_label)
        weights_layout.addLayout(l_layout)
        
        # a weight (Green-Red)
        a_layout = QHBoxLayout()
        a_name = QLabel("a (G-R):")
        a_name.setFixedWidth(70)
        self._a_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self._a_weight_slider.setRange(0, 100)
        self._a_weight_slider.setValue(30)
        self._a_weight_slider.setToolTip("Green-Red axis weight (0-100%)")
        self._a_weight_slider.valueChanged.connect(self._on_a_weight_changed)
        self._a_weight_label = QLabel("30%")
        self._a_weight_label.setFixedWidth(35)
        a_layout.addWidget(a_name)
        a_layout.addWidget(self._a_weight_slider, 1)
        a_layout.addWidget(self._a_weight_label)
        weights_layout.addLayout(a_layout)
        
        # b weight (Blue-Yellow)
        b_layout = QHBoxLayout()
        b_name = QLabel("b (B-Y):")
        b_name.setFixedWidth(70)
        self._b_weight_slider = QSlider(Qt.Orientation.Horizontal)
        self._b_weight_slider.setRange(0, 100)
        self._b_weight_slider.setValue(30)
        self._b_weight_slider.setToolTip("Blue-Yellow axis weight (0-100%)")
        self._b_weight_slider.valueChanged.connect(self._on_b_weight_changed)
        self._b_weight_label = QLabel("30%")
        self._b_weight_label.setFixedWidth(35)
        b_layout.addWidget(b_name)
        b_layout.addWidget(self._b_weight_slider, 1)
        b_layout.addWidget(self._b_weight_label)
        weights_layout.addLayout(b_layout)
        
        layout.addWidget(weights_group)
        
        # Achromatic Handling group
        achro_group = QGroupBox("Achromatic Handling")
        achro_layout = QVBoxLayout(achro_group)
        achro_layout.setSpacing(4)
        
        # Achromatic threshold
        thresh_layout = QHBoxLayout()
        thresh_label = QLabel("Threshold:")
        thresh_label.setFixedWidth(70)
        self._achromatic_threshold_spin = QSpinBox()
        self._achromatic_threshold_spin.setRange(0, 50)
        self._achromatic_threshold_spin.setValue(10)
        self._achromatic_threshold_spin.setToolTip(
            "Colors with chroma (sqrt(a^2+b^2)) below this\n"
            "are treated as grayscale."
        )
        self._achromatic_threshold_spin.valueChanged.connect(self._on_option_changed)
        thresh_layout.addWidget(thresh_label)
        thresh_layout.addWidget(self._achromatic_threshold_spin, 1)
        achro_layout.addLayout(thresh_layout)
        
        # Preserve black/white
        self._preserve_bw_check = QCheckBox("Preserve pure black and white")
        self._preserve_bw_check.setChecked(True)
        self._preserve_bw_check.setToolTip(
            "Keep pure black (0,0,0) and white (255,255,255)\n"
            "as separate palette entries."
        )
        self._preserve_bw_check.stateChanged.connect(self._on_option_changed)
        achro_layout.addWidget(self._preserve_bw_check)
        
        layout.addWidget(achro_group)
        
        # Color Snapping group
        snap_group = QGroupBox("Color Snapping (Optional)")
        snap_layout = QVBoxLayout(snap_group)
        snap_layout.setSpacing(4)
        
        self._snap_colors_check = QCheckBox("Snap colors to clean values")
        self._snap_colors_check.setChecked(False)
        self._snap_colors_check.setToolTip(
            "Round final palette colors to clean LAB values.\n"
            "Creates more uniform, \"designed\" looking palettes."
        )
        self._snap_colors_check.stateChanged.connect(self._on_snap_changed)
        snap_layout.addWidget(self._snap_colors_check)
        
        snap_values_layout = QHBoxLayout()
        
        snap_l_label = QLabel("L step:")
        snap_l_label.setFixedWidth(50)
        self._snap_l_spin = QSpinBox()
        self._snap_l_spin.setRange(1, 50)
        self._snap_l_spin.setValue(10)
        self._snap_l_spin.setEnabled(False)
        self._snap_l_spin.setToolTip("Lightness snapping step")
        self._snap_l_spin.valueChanged.connect(self._on_option_changed)
        
        snap_ab_label = QLabel("a/b step:")
        snap_ab_label.setFixedWidth(50)
        self._snap_ab_spin = QSpinBox()
        self._snap_ab_spin.setRange(1, 50)
        self._snap_ab_spin.setValue(10)
        self._snap_ab_spin.setEnabled(False)
        self._snap_ab_spin.setToolTip("Color axis snapping step")
        self._snap_ab_spin.valueChanged.connect(self._on_option_changed)
        
        snap_values_layout.addWidget(snap_l_label)
        snap_values_layout.addWidget(self._snap_l_spin)
        snap_values_layout.addWidget(snap_ab_label)
        snap_values_layout.addWidget(self._snap_ab_spin)
        snap_layout.addLayout(snap_values_layout)
        
        layout.addWidget(snap_group)
        
        # Spacer
        layout.addStretch()
        
        return self._widget
    
    def _on_l_weight_changed(self, value: int):
        """Handle L weight slider change"""
        if self._l_weight_label:
            self._l_weight_label.setText(f"{value}%")
        self._notify_change()
    
    def _on_a_weight_changed(self, value: int):
        """Handle a weight slider change"""
        if self._a_weight_label:
            self._a_weight_label.setText(f"{value}%")
        self._notify_change()
    
    def _on_b_weight_changed(self, value: int):
        """Handle b weight slider change"""
        if self._b_weight_label:
            self._b_weight_label.setText(f"{value}%")
        self._notify_change()
    
    def _on_snap_changed(self, state: int):
        """Handle snap checkbox change"""
        enabled = state == Qt.CheckState.Checked.value
        if self._snap_l_spin:
            self._snap_l_spin.setEnabled(enabled)
        if self._snap_ab_spin:
            self._snap_ab_spin.setEnabled(enabled)
        self._notify_change()
    
    def _on_option_changed(self, *args):
        """Handle generic option change"""
        self._notify_change()
    
    def _notify_change(self):
        """Notify that parameters changed"""
        if self._callback:
            self._callback()
    
    def _get_weights(self) -> tuple:
        """Get normalized LAB weights"""
        l_w = self._l_weight_slider.value() if self._l_weight_slider else 40
        a_w = self._a_weight_slider.value() if self._a_weight_slider else 30
        b_w = self._b_weight_slider.value() if self._b_weight_slider else 30
        
        total = l_w + a_w + b_w
        if total == 0:
            return (1.0, 1.0, 1.0)
        
        return (l_w / total, a_w / total, b_w / total)
    
    def _snap_lab_value(self, lab_colors: np.ndarray) -> np.ndarray:
        """Snap LAB colors to clean values"""
        if not self._snap_colors_check or not self._snap_colors_check.isChecked():
            return lab_colors
        
        l_step = self._snap_l_spin.value() if self._snap_l_spin else 10
        ab_step = self._snap_ab_spin.value() if self._snap_ab_spin else 10
        
        result = lab_colors.copy()
        # L channel (0-255 in OpenCV's LAB)
        result[:, 0] = np.round(result[:, 0] / l_step) * l_step
        # a channel (0-255 in OpenCV's LAB, 128 is neutral)
        result[:, 1] = np.round(result[:, 1] / ab_step) * ab_step
        # b channel (0-255 in OpenCV's LAB, 128 is neutral)
        result[:, 2] = np.round(result[:, 2] / ab_step) * ab_step
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Process the image with color quantization"""
        if image is None:
            return None
        
        if self._num_colors_spin is None:
            return image
        
        # Get parameters
        num_colors = self._num_colors_spin.value()
        iterations = self._iterations_spin.value() if self._iterations_spin else 10
        achro_threshold = self._achromatic_threshold_spin.value() if self._achromatic_threshold_spin else 10
        preserve_bw = self._preserve_bw_check.isChecked() if self._preserve_bw_check else True
        
        # Separate alpha channel if present
        has_alpha = len(image.shape) == 3 and image.shape[2] == 4
        if has_alpha:
            bgr = image[:, :, :3]
            alpha = image[:, :, 3]
        else:
            bgr = image
            alpha = None
        
        # Convert to LAB
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        
        h, w = lab.shape[:2]
        
        # Create mask for pixels to process (non-transparent)
        if alpha is not None:
            valid_mask = alpha > 0
        else:
            valid_mask = np.ones((h, w), dtype=bool)
        
        # Get valid pixels
        valid_pixels_lab = lab[valid_mask]
        
        if len(valid_pixels_lab) == 0:
            return image.copy()
        
        # Identify achromatic pixels (low chroma in LAB)
        # In OpenCV LAB: L is 0-255, a and b are 0-255 with 128 being neutral
        a_centered = valid_pixels_lab[:, 1].astype(float) - 128
        b_centered = valid_pixels_lab[:, 2].astype(float) - 128
        chroma = np.sqrt(a_centered**2 + b_centered**2)
        
        is_achromatic = chroma < achro_threshold
        chromatic_pixels = valid_pixels_lab[~is_achromatic]
        achromatic_pixels = valid_pixels_lab[is_achromatic]
        
        # Determine color allocation
        reserved_colors = 0
        if preserve_bw:
            reserved_colors = 2  # Black and white
        
        # Allocate colors proportionally
        total_pixels = len(valid_pixels_lab)
        chromatic_ratio = len(chromatic_pixels) / total_pixels if total_pixels > 0 else 0.5
        
        chromatic_colors = max(1, int((num_colors - reserved_colors) * chromatic_ratio))
        achromatic_colors = max(1, num_colors - reserved_colors - chromatic_colors)
        
        # Adjust if we have no pixels of one type
        if len(chromatic_pixels) == 0:
            achromatic_colors = num_colors - reserved_colors
            chromatic_colors = 0
        elif len(achromatic_pixels) == 0:
            chromatic_colors = num_colors - reserved_colors
            achromatic_colors = 0
        
        # Get weights
        l_weight, a_weight, b_weight = self._get_weights()
        
        palette = []
        
        # Add reserved black and white
        if preserve_bw:
            # Black in LAB (L=0)
            palette.append(np.array([0, 128, 128], dtype=np.uint8))
            # White in LAB (L=255)
            palette.append(np.array([255, 128, 128], dtype=np.uint8))
        
        # Cluster chromatic pixels
        if chromatic_colors > 0 and len(chromatic_pixels) > 0:
            # Apply weights for clustering
            weighted_chromatic = chromatic_pixels.astype(np.float32).copy()
            weighted_chromatic[:, 0] *= l_weight * 3  # Scale factor for balance
            weighted_chromatic[:, 1] *= a_weight * 3
            weighted_chromatic[:, 2] *= b_weight * 3
            
            # K-Means clustering
            k = min(chromatic_colors, len(chromatic_pixels))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, 1.0)
            _, labels, centers = cv2.kmeans(
                weighted_chromatic,
                k,
                None,
                criteria,
                attempts=3,
                flags=cv2.KMEANS_PP_CENTERS
            )
            
            # Convert centers back to unweighted LAB
            centers[:, 0] /= (l_weight * 3) if l_weight > 0 else 1
            centers[:, 1] /= (a_weight * 3) if a_weight > 0 else 1
            centers[:, 2] /= (b_weight * 3) if b_weight > 0 else 1
            centers = np.clip(centers, 0, 255).astype(np.uint8)
            
            # Apply snapping if enabled
            centers = self._snap_lab_value(centers)
            
            for center in centers:
                palette.append(center)
        
        # Cluster achromatic pixels (by L only)
        if achromatic_colors > 0 and len(achromatic_pixels) > 0:
            # For achromatic, only cluster by L value
            l_values = achromatic_pixels[:, 0:1].astype(np.float32)
            
            k = min(achromatic_colors, len(achromatic_pixels))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, 1.0)
            _, labels, centers = cv2.kmeans(
                l_values,
                k,
                None,
                criteria,
                attempts=3,
                flags=cv2.KMEANS_PP_CENTERS
            )
            
            # Create neutral LAB colors (a=128, b=128)
            for center in centers:
                l_val = int(np.clip(center[0], 0, 255))
                if self._snap_colors_check and self._snap_colors_check.isChecked():
                    l_step = self._snap_l_spin.value() if self._snap_l_spin else 10
                    l_val = int(round(l_val / l_step) * l_step)
                    l_val = int(np.clip(l_val, 0, 255))
                palette.append(np.array([l_val, 128, 128], dtype=np.uint8))
        
        # Convert palette to numpy array
        if len(palette) == 0:
            return image.copy()
        
        palette_array = np.array(palette, dtype=np.uint8)
        
        # Map all pixels to nearest palette color
        result_lab = lab.copy()
        
        # Process only valid pixels
        valid_lab = lab[valid_mask].astype(np.float32)
        palette_float = palette_array.astype(np.float32)
        
        # Apply weights for distance calculation
        weighted_valid = valid_lab.copy()
        weighted_valid[:, 0] *= l_weight
        weighted_valid[:, 1] *= a_weight
        weighted_valid[:, 2] *= b_weight
        
        weighted_palette = palette_float.copy()
        weighted_palette[:, 0] *= l_weight
        weighted_palette[:, 1] *= a_weight
        weighted_palette[:, 2] *= b_weight
        
        # Find nearest palette color for each pixel
        # Using broadcasting for efficiency
        distances = np.zeros((len(valid_lab), len(palette_array)), dtype=np.float32)
        for i, wp in enumerate(weighted_palette):
            diff = weighted_valid - wp
            distances[:, i] = np.sum(diff**2, axis=1)
        
        nearest_indices = np.argmin(distances, axis=1)
        quantized_pixels = palette_array[nearest_indices]
        
        # Put quantized pixels back
        result_lab[valid_mask] = quantized_pixels
        
        # Convert back to BGR
        result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        # Restore alpha channel
        if has_alpha:
            result = np.zeros_like(image)
            result[:, :, :3] = result_bgr
            result[:, :, 3] = alpha
            return result
        else:
            return result_bgr
    
    def on_parameters_changed(self, callback: Callable[[], None]) -> None:
        """Register callback for parameter changes"""
        self._callback = callback
    
    def save_settings(self, config: ConfigManager) -> None:
        """Save operator settings"""
        if self._num_colors_spin:
            config.save_operator_setting(self.name, "num_colors", self._num_colors_spin.value())
        if self._iterations_spin:
            config.save_operator_setting(self.name, "iterations", self._iterations_spin.value())
        if self._l_weight_slider:
            config.save_operator_setting(self.name, "l_weight", self._l_weight_slider.value())
        if self._a_weight_slider:
            config.save_operator_setting(self.name, "a_weight", self._a_weight_slider.value())
        if self._b_weight_slider:
            config.save_operator_setting(self.name, "b_weight", self._b_weight_slider.value())
        if self._achromatic_threshold_spin:
            config.save_operator_setting(self.name, "achromatic_threshold", self._achromatic_threshold_spin.value())
        if self._preserve_bw_check:
            config.save_operator_setting(self.name, "preserve_bw", self._preserve_bw_check.isChecked())
        if self._snap_colors_check:
            config.save_operator_setting(self.name, "snap_colors", self._snap_colors_check.isChecked())
        if self._snap_l_spin:
            config.save_operator_setting(self.name, "snap_l", self._snap_l_spin.value())
        if self._snap_ab_spin:
            config.save_operator_setting(self.name, "snap_ab", self._snap_ab_spin.value())
    
    def load_settings(self, config: ConfigManager) -> None:
        """Load operator settings"""
        # Ensure widget is created
        self.get_widget()
        
        num_colors = config.load_operator_setting(self.name, "num_colors", 16)
        if self._num_colors_spin:
            if isinstance(num_colors, str):
                num_colors = int(num_colors)
            self._num_colors_spin.setValue(int(num_colors))
        
        iterations = config.load_operator_setting(self.name, "iterations", 10)
        if self._iterations_spin:
            if isinstance(iterations, str):
                iterations = int(iterations)
            self._iterations_spin.setValue(int(iterations))
        
        l_weight = config.load_operator_setting(self.name, "l_weight", 40)
        if self._l_weight_slider:
            if isinstance(l_weight, str):
                l_weight = int(l_weight)
            self._l_weight_slider.setValue(int(l_weight))
            self._l_weight_label.setText(f"{l_weight}%")
        
        a_weight = config.load_operator_setting(self.name, "a_weight", 30)
        if self._a_weight_slider:
            if isinstance(a_weight, str):
                a_weight = int(a_weight)
            self._a_weight_slider.setValue(int(a_weight))
            self._a_weight_label.setText(f"{a_weight}%")
        
        b_weight = config.load_operator_setting(self.name, "b_weight", 30)
        if self._b_weight_slider:
            if isinstance(b_weight, str):
                b_weight = int(b_weight)
            self._b_weight_slider.setValue(int(b_weight))
            self._b_weight_label.setText(f"{b_weight}%")
        
        achro_threshold = config.load_operator_setting(self.name, "achromatic_threshold", 10)
        if self._achromatic_threshold_spin:
            if isinstance(achro_threshold, str):
                achro_threshold = int(achro_threshold)
            self._achromatic_threshold_spin.setValue(int(achro_threshold))
        
        preserve_bw = config.load_operator_setting(self.name, "preserve_bw", True)
        if self._preserve_bw_check:
            if isinstance(preserve_bw, str):
                preserve_bw = preserve_bw.lower() == "true"
            self._preserve_bw_check.setChecked(bool(preserve_bw))
        
        snap_colors = config.load_operator_setting(self.name, "snap_colors", False)
        if self._snap_colors_check:
            if isinstance(snap_colors, str):
                snap_colors = snap_colors.lower() == "true"
            self._snap_colors_check.setChecked(bool(snap_colors))
        
        snap_l = config.load_operator_setting(self.name, "snap_l", 10)
        if self._snap_l_spin:
            if isinstance(snap_l, str):
                snap_l = int(snap_l)
            self._snap_l_spin.setValue(int(snap_l))
            self._snap_l_spin.setEnabled(self._snap_colors_check.isChecked() if self._snap_colors_check else False)
        
        snap_ab = config.load_operator_setting(self.name, "snap_ab", 10)
        if self._snap_ab_spin:
            if isinstance(snap_ab, str):
                snap_ab = int(snap_ab)
            self._snap_ab_spin.setValue(int(snap_ab))
            self._snap_ab_spin.setEnabled(self._snap_colors_check.isChecked() if self._snap_colors_check else False)
    
    def reset_to_defaults(self) -> None:
        """Reset to default values"""
        if self._num_colors_spin:
            self._num_colors_spin.setValue(16)
        if self._iterations_spin:
            self._iterations_spin.setValue(10)
        if self._l_weight_slider:
            self._l_weight_slider.setValue(40)
            self._l_weight_label.setText("40%")
        if self._a_weight_slider:
            self._a_weight_slider.setValue(30)
            self._a_weight_label.setText("30%")
        if self._b_weight_slider:
            self._b_weight_slider.setValue(30)
            self._b_weight_label.setText("30%")
        if self._achromatic_threshold_spin:
            self._achromatic_threshold_spin.setValue(10)
        if self._preserve_bw_check:
            self._preserve_bw_check.setChecked(True)
        if self._snap_colors_check:
            self._snap_colors_check.setChecked(False)
        if self._snap_l_spin:
            self._snap_l_spin.setValue(10)
            self._snap_l_spin.setEnabled(False)
        if self._snap_ab_spin:
            self._snap_ab_spin.setValue(10)
            self._snap_ab_spin.setEnabled(False)

