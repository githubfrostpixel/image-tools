"""
Add Border Operator - Add hugging borders around non-transparent pixels
"""
import cv2
import numpy as np
from typing import Callable, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QCheckBox, QGroupBox, QLineEdit
)
from PyQt6.QtCore import Qt

from core.operator_base import OperatorBase
from core.config_manager import ConfigManager


class AddBorderOperator(OperatorBase):
    """
    Operator for adding hugging borders around non-transparent pixels.
    
    Features:
    - Black border (inner)
    - White border (outer)
    - Configurable offset for shadow effect
    - Customizable border colors
    """
    
    def __init__(self):
        self._widget: Optional[QWidget] = None
        
        # Border options
        self._black_border_check: Optional[QCheckBox] = None
        self._white_border_check: Optional[QCheckBox] = None
        
        # Offset options
        self._offset_x_spin: Optional[QSpinBox] = None
        self._offset_y_spin: Optional[QSpinBox] = None
        
        # Color options
        self._inner_color_edit: Optional[QLineEdit] = None
        self._outer_color_edit: Optional[QLineEdit] = None
        
        self._callback: Optional[Callable] = None
    
    @property
    def name(self) -> str:
        return "Add Border"
    
    def get_widget(self) -> QWidget:
        """Create and return the parameter widget"""
        if self._widget is not None:
            return self._widget
        
        self._widget = QWidget()
        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        
        # Border Style group
        style_group = QGroupBox("Border Style")
        style_layout = QVBoxLayout(style_group)
        style_layout.setSpacing(4)
        
        self._black_border_check = QCheckBox("Black border (inner)")
        self._black_border_check.setChecked(True)
        self._black_border_check.setToolTip(
            "Add a 1-pixel black border hugging the non-transparent pixels."
        )
        self._black_border_check.stateChanged.connect(self._on_option_changed)
        style_layout.addWidget(self._black_border_check)
        
        self._white_border_check = QCheckBox("White border (outer)")
        self._white_border_check.setChecked(False)
        self._white_border_check.setToolTip(
            "Add a 1-pixel white border around the black border."
        )
        self._white_border_check.stateChanged.connect(self._on_option_changed)
        style_layout.addWidget(self._white_border_check)
        
        layout.addWidget(style_group)
        
        # Offset group
        offset_group = QGroupBox("Offset")
        offset_layout = QVBoxLayout(offset_group)
        offset_layout.setSpacing(4)
        
        offset_desc = QLabel("Shift original image relative to borders (shadow effect)")
        offset_desc.setStyleSheet("color: #666; font-size: 10px;")
        offset_desc.setWordWrap(True)
        offset_layout.addWidget(offset_desc)
        
        xy_layout = QHBoxLayout()
        
        x_label = QLabel("X:")
        x_label.setFixedWidth(20)
        self._offset_x_spin = QSpinBox()
        self._offset_x_spin.setRange(-10, 10)
        self._offset_x_spin.setValue(0)
        self._offset_x_spin.setSuffix(" px")
        self._offset_x_spin.valueChanged.connect(self._on_option_changed)
        
        y_label = QLabel("Y:")
        y_label.setFixedWidth(20)
        self._offset_y_spin = QSpinBox()
        self._offset_y_spin.setRange(-10, 10)
        self._offset_y_spin.setValue(0)
        self._offset_y_spin.setSuffix(" px")
        self._offset_y_spin.valueChanged.connect(self._on_option_changed)
        
        xy_layout.addWidget(x_label)
        xy_layout.addWidget(self._offset_x_spin, 1)
        xy_layout.addWidget(y_label)
        xy_layout.addWidget(self._offset_y_spin, 1)
        offset_layout.addLayout(xy_layout)
        
        layout.addWidget(offset_group)
        
        # Colors group
        colors_group = QGroupBox("Colors")
        colors_layout = QVBoxLayout(colors_group)
        colors_layout.setSpacing(4)
        
        inner_layout = QHBoxLayout()
        inner_label = QLabel("Inner:")
        inner_label.setFixedWidth(50)
        self._inner_color_edit = QLineEdit("#000000")
        self._inner_color_edit.setToolTip(
            "Inner border color: hex (#RRGGBB), name (black, white), or R,G,B"
        )
        self._inner_color_edit.textChanged.connect(self._on_option_changed)
        inner_layout.addWidget(inner_label)
        inner_layout.addWidget(self._inner_color_edit, 1)
        colors_layout.addLayout(inner_layout)
        
        outer_layout = QHBoxLayout()
        outer_label = QLabel("Outer:")
        outer_label.setFixedWidth(50)
        self._outer_color_edit = QLineEdit("#FFFFFF")
        self._outer_color_edit.setToolTip(
            "Outer border color: hex (#RRGGBB), name (black, white), or R,G,B"
        )
        self._outer_color_edit.textChanged.connect(self._on_option_changed)
        outer_layout.addWidget(outer_label)
        outer_layout.addWidget(self._outer_color_edit, 1)
        colors_layout.addLayout(outer_layout)
        
        layout.addWidget(colors_group)
        
        # Spacer
        layout.addStretch()
        
        return self._widget
    
    def _on_option_changed(self, *args):
        """Handle option change"""
        self._notify_change()
    
    def _notify_change(self):
        """Notify that parameters changed"""
        if self._callback:
            self._callback()
    
    @staticmethod
    def _parse_color(color_str: str) -> tuple:
        """Parse color string to RGBA tuple"""
        if color_str is None:
            return (0, 0, 0, 255)
        color_str = color_str.strip().lower()
        if color_str == 'black':
            return (0, 0, 0, 255)
        if color_str == 'white':
            return (255, 255, 255, 255)
        if color_str.startswith('#') and len(color_str) == 7:
            try:
                r = int(color_str[1:3], 16)
                g = int(color_str[3:5], 16)
                b = int(color_str[5:7], 16)
                return (r, g, b, 255)
            except ValueError:
                return (0, 0, 0, 255)
        if ',' in color_str:
            parts = color_str.split(',')
            if len(parts) >= 3:
                try:
                    r = int(parts[0].strip())
                    g = int(parts[1].strip())
                    b = int(parts[2].strip())
                    a = int(parts[3].strip()) if len(parts) > 3 else 255
                    return (r, g, b, a)
                except ValueError:
                    return (0, 0, 0, 255)
        return (0, 0, 0, 255)
    
    @staticmethod
    def _add_hugging_border(image: np.ndarray, border_color: tuple) -> np.ndarray:
        """
        Add a 1-pixel hugging border around non-transparent pixels.
        
        Args:
            image: Input image in BGRA format
            border_color: Border color as (R, G, B, A) tuple
            
        Returns:
            New image with border added (canvas expanded by 2 pixels each direction)
        """
        if len(image.shape) != 3 or image.shape[2] != 4:
            return image
        
        h, w = image.shape[:2]
        alpha = image[:, :, 3]
        
        # Find non-transparent pixels
        non_transparent = alpha > 0
        if not np.any(non_transparent):
            return image.copy()
        
        # Find bounding box of non-transparent region
        coords = np.where(non_transparent)
        min_y, max_y = coords[0].min(), coords[0].max()
        min_x, max_x = coords[1].min(), coords[1].max()
        
        # Crop to non-transparent region
        cropped = image[min_y:max_y+1, min_x:max_x+1].copy()
        crop_h, crop_w = cropped.shape[:2]
        
        # Create new canvas with 1-pixel border on each side
        bordered = np.zeros((crop_h + 2, crop_w + 2, 4), dtype=np.uint8)
        
        # Place cropped image in center
        bordered[1:-1, 1:-1] = cropped
        
        # Create mask of opaque pixels in cropped image
        mask = cropped[:, :, 3] > 0
        
        # Add border pixels in 4 directions (up, down, left, right)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            y_indices, x_indices = np.where(mask)
            # Offset by 1 for the border canvas, then apply direction
            yb = y_indices + 1 + dy
            xb = x_indices + 1 + dx
            
            # Check bounds
            valid = (yb >= 0) & (yb < crop_h + 2) & (xb >= 0) & (xb < crop_w + 2)
            yb = yb[valid]
            xb = xb[valid]
            
            # Only set border where there's no existing content
            empty_mask = bordered[yb, xb, 3] == 0
            yb = yb[empty_mask]
            xb = xb[empty_mask]
            
            # Set border color (BGRA format)
            bordered[yb, xb, 0] = border_color[2]  # B
            bordered[yb, xb, 1] = border_color[1]  # G
            bordered[yb, xb, 2] = border_color[0]  # R
            bordered[yb, xb, 3] = border_color[3]  # A
        
        return bordered
    
    @staticmethod
    def _composite_layers(
        original: np.ndarray,
        inner_border: np.ndarray,
        outer_border: Optional[np.ndarray],
        offset_x: int,
        offset_y: int
    ) -> np.ndarray:
        """
        Composite layers together with offset.
        
        Args:
            original: Original image (BGRA)
            inner_border: Image with inner border (BGRA)
            outer_border: Image with outer border (BGRA), or None
            offset_x: X offset for original relative to borders
            offset_y: Y offset for original relative to borders
            
        Returns:
            Composited image (BGRA)
        """
        if outer_border is not None:
            # Start with outer border as base
            base = outer_border
        else:
            # Start with inner border as base
            base = inner_border
        
        base_h, base_w = base.shape[:2]
        result = base.copy()
        
        # If we have outer border, composite inner border onto it (centered)
        if outer_border is not None:
            inner_h, inner_w = inner_border.shape[:2]
            # Inner border is 2 pixels smaller than outer in each dimension
            # Center it
            ix = (base_w - inner_w) // 2
            iy = (base_h - inner_h) // 2
            
            # Alpha composite inner onto result
            for c in range(3):
                inner_alpha = inner_border[:, :, 3].astype(float) / 255.0
                for y in range(inner_h):
                    for x in range(inner_w):
                        if inner_border[y, x, 3] > 0:
                            ty, tx = iy + y, ix + x
                            if 0 <= ty < base_h and 0 <= tx < base_w:
                                a = inner_alpha[y, x]
                                result[ty, tx, c] = int(
                                    inner_border[y, x, c] * a + 
                                    result[ty, tx, c] * (1 - a)
                                )
                                result[ty, tx, 3] = max(result[ty, tx, 3], inner_border[y, x, 3])
        
        # Composite original onto result with offset
        orig_h, orig_w = original.shape[:2]
        
        # Calculate center position for original, then apply offset
        # The inner border is 2 pixels larger than original (1 on each side)
        # The outer border (if present) is 2 more pixels larger
        if outer_border is not None:
            # Original should be centered in the outer border canvas
            # Outer is 4 pixels larger than original (2 for inner border, 2 for outer)
            ox = (base_w - orig_w) // 2 + offset_x
            oy = (base_h - orig_h) // 2 + offset_y
        else:
            # Original should be centered in the inner border canvas
            # Inner is 2 pixels larger than original
            ox = (base_w - orig_w) // 2 + offset_x
            oy = (base_h - orig_h) // 2 + offset_y
        
        # Alpha composite original onto result
        for y in range(orig_h):
            for x in range(orig_w):
                if original[y, x, 3] > 0:
                    ty, tx = oy + y, ox + x
                    if 0 <= ty < base_h and 0 <= tx < base_w:
                        a = original[y, x, 3] / 255.0
                        for c in range(3):
                            result[ty, tx, c] = int(
                                original[y, x, c] * a + 
                                result[ty, tx, c] * (1 - a)
                            )
                        result[ty, tx, 3] = max(result[ty, tx, 3], original[y, x, 3])
        
        return result
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Process the image with border addition"""
        if image is None:
            return None
        
        # Check if any border is enabled
        black_enabled = self._black_border_check and self._black_border_check.isChecked()
        white_enabled = self._white_border_check and self._white_border_check.isChecked()
        
        if not black_enabled and not white_enabled:
            # No border selected, return original
            return image.copy()
        
        # Get colors
        inner_color = self._parse_color(
            self._inner_color_edit.text() if self._inner_color_edit else "#000000"
        )
        outer_color = self._parse_color(
            self._outer_color_edit.text() if self._outer_color_edit else "#FFFFFF"
        )
        
        # Get offset
        offset_x = self._offset_x_spin.value() if self._offset_x_spin else 0
        offset_y = self._offset_y_spin.value() if self._offset_y_spin else 0
        
        # Crop to non-transparent region first
        alpha = image[:, :, 3]
        non_transparent = alpha > 0
        if not np.any(non_transparent):
            return image.copy()
        
        coords = np.where(non_transparent)
        min_y, max_y = coords[0].min(), coords[0].max()
        min_x, max_x = coords[1].min(), coords[1].max()
        original_cropped = image[min_y:max_y+1, min_x:max_x+1].copy()
        
        # Build border layers
        inner_border = None
        outer_border = None
        
        if black_enabled:
            inner_border = self._add_hugging_border(original_cropped, inner_color)
            
            if white_enabled:
                outer_border = self._add_hugging_border(inner_border, outer_color)
        elif white_enabled:
            # Only white border (no black)
            inner_border = self._add_hugging_border(original_cropped, outer_color)
        
        # Composite layers
        if inner_border is not None:
            result = self._composite_layers(
                original_cropped,
                inner_border,
                outer_border,
                offset_x,
                offset_y
            )
            return result
        
        return image.copy()
    
    def on_parameters_changed(self, callback: Callable[[], None]) -> None:
        """Register callback for parameter changes"""
        self._callback = callback
    
    def save_settings(self, config: ConfigManager) -> None:
        """Save operator settings"""
        if self._black_border_check:
            config.save_operator_setting(self.name, "black_border", self._black_border_check.isChecked())
        if self._white_border_check:
            config.save_operator_setting(self.name, "white_border", self._white_border_check.isChecked())
        if self._offset_x_spin:
            config.save_operator_setting(self.name, "offset_x", self._offset_x_spin.value())
        if self._offset_y_spin:
            config.save_operator_setting(self.name, "offset_y", self._offset_y_spin.value())
        if self._inner_color_edit:
            config.save_operator_setting(self.name, "inner_color", self._inner_color_edit.text())
        if self._outer_color_edit:
            config.save_operator_setting(self.name, "outer_color", self._outer_color_edit.text())
    
    def load_settings(self, config: ConfigManager) -> None:
        """Load operator settings"""
        # Ensure widget is created
        self.get_widget()
        
        black_border = config.load_operator_setting(self.name, "black_border", True)
        if self._black_border_check:
            if isinstance(black_border, str):
                black_border = black_border.lower() == "true"
            self._black_border_check.setChecked(bool(black_border))
        
        white_border = config.load_operator_setting(self.name, "white_border", False)
        if self._white_border_check:
            if isinstance(white_border, str):
                white_border = white_border.lower() == "true"
            self._white_border_check.setChecked(bool(white_border))
        
        offset_x = config.load_operator_setting(self.name, "offset_x", 0)
        if self._offset_x_spin:
            if isinstance(offset_x, str):
                offset_x = int(offset_x)
            self._offset_x_spin.setValue(int(offset_x))
        
        offset_y = config.load_operator_setting(self.name, "offset_y", 0)
        if self._offset_y_spin:
            if isinstance(offset_y, str):
                offset_y = int(offset_y)
            self._offset_y_spin.setValue(int(offset_y))
        
        inner_color = config.load_operator_setting(self.name, "inner_color", "#000000")
        if self._inner_color_edit:
            self._inner_color_edit.setText(str(inner_color))
        
        outer_color = config.load_operator_setting(self.name, "outer_color", "#FFFFFF")
        if self._outer_color_edit:
            self._outer_color_edit.setText(str(outer_color))
    
    def reset_to_defaults(self) -> None:
        """Reset to default values"""
        if self._black_border_check:
            self._black_border_check.setChecked(True)
        if self._white_border_check:
            self._white_border_check.setChecked(False)
        if self._offset_x_spin:
            self._offset_x_spin.setValue(0)
        if self._offset_y_spin:
            self._offset_y_spin.setValue(0)
        if self._inner_color_edit:
            self._inner_color_edit.setText("#000000")
        if self._outer_color_edit:
            self._outer_color_edit.setText("#FFFFFF")

