"""
Pixel Snapper Operator - Snap messy pixel art to a perfect grid
"""
from dataclasses import replace
from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QCheckBox, QGroupBox, QComboBox, QDoubleSpinBox
)

from core.config_manager import ConfigManager
from core.operator_base import OperatorBase
from core.pixel_snapper_algo import SnapperConfig, process_pixel_snap


class PixelSnapperOperator(OperatorBase):
    """Snap pixels to a detected grid."""

    OUTPUT_DOWNSAMPLE = "Downsample to grid"
    OUTPUT_INPLACE = "Snap in place"

    def __init__(self):
        self._widget: Optional[QWidget] = None
        self._callback: Optional[Callable] = None
        self._source_width = 0
        self._source_height = 0

        self._snapping_check: Optional[QCheckBox] = None
        self._pixel_size_spin: Optional[QSpinBox] = None
        self._output_mode_combo: Optional[QComboBox] = None
        self._upscale_spin: Optional[QSpinBox] = None

        self._advanced_group: Optional[QGroupBox] = None
        self._peak_threshold_spin: Optional[QDoubleSpinBox] = None
        self._peak_distance_spin: Optional[QSpinBox] = None
        self._walker_window_spin: Optional[QDoubleSpinBox] = None
        self._walker_min_window_spin: Optional[QDoubleSpinBox] = None
        self._walker_strength_spin: Optional[QDoubleSpinBox] = None
        self._min_cuts_spin: Optional[QSpinBox] = None
        self._fallback_segments_spin: Optional[QSpinBox] = None
        self._max_step_ratio_spin: Optional[QDoubleSpinBox] = None

    @property
    def name(self) -> str:
        return "Pixel Snapper"

    def get_widget(self) -> QWidget:
        if self._widget is not None:
            return self._widget

        self._widget = QWidget()
        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        grid_group = QGroupBox("Grid Detection")
        grid_layout = QVBoxLayout(grid_group)

        self._snapping_check = QCheckBox("Enable grid snapping")
        self._snapping_check.setChecked(True)
        self._snapping_check.setToolTip("Snap pixels to a detected grid.")
        self._snapping_check.stateChanged.connect(self._on_snapping_changed)
        grid_layout.addWidget(self._snapping_check)

        pixel_layout = QHBoxLayout()
        pixel_label = QLabel("Pixel size:")
        pixel_label.setFixedWidth(70)
        self._pixel_size_spin = QSpinBox()
        self._pixel_size_spin.setRange(0, 512)
        self._pixel_size_spin.setValue(0)
        self._pixel_size_spin.setSpecialValueText("Auto")
        self._pixel_size_spin.setToolTip(
            "Force grid cell size in pixels.\n"
            "0 = auto-detect from image edges."
        )
        self._pixel_size_spin.valueChanged.connect(self._on_option_changed)
        pixel_layout.addWidget(pixel_label)
        pixel_layout.addWidget(self._pixel_size_spin, 1)
        grid_layout.addLayout(pixel_layout)

        layout.addWidget(grid_group)

        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)

        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        mode_label.setFixedWidth(70)
        self._output_mode_combo = QComboBox()
        self._output_mode_combo.addItems([self.OUTPUT_DOWNSAMPLE, self.OUTPUT_INPLACE])
        self._output_mode_combo.currentIndexChanged.connect(self._on_output_mode_changed)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self._output_mode_combo, 1)
        output_layout.addLayout(mode_layout)

        upscale_layout = QHBoxLayout()
        upscale_label = QLabel("Upscale:")
        upscale_label.setFixedWidth(70)
        self._upscale_spin = QSpinBox()
        self._upscale_spin.setRange(1, 16)
        self._upscale_spin.setValue(1)
        self._upscale_spin.setToolTip("Integer nearest-neighbor upscale after downsampling.")
        self._upscale_spin.valueChanged.connect(self._on_option_changed)
        upscale_layout.addWidget(upscale_label)
        upscale_layout.addWidget(self._upscale_spin, 1)
        output_layout.addLayout(upscale_layout)

        layout.addWidget(output_group)

        self._advanced_group = QGroupBox("Advanced")
        self._advanced_group.setCheckable(True)
        self._advanced_group.setChecked(False)
        advanced_layout = QVBoxLayout(self._advanced_group)

        self._peak_threshold_spin = self._add_double_spin(
            advanced_layout, "Peak threshold:", 0.01, 1.0, 0.2, 0.05
        )
        self._peak_distance_spin = self._add_int_spin(
            advanced_layout, "Peak distance:", 1, 32, 4
        )
        self._walker_window_spin = self._add_double_spin(
            advanced_layout, "Walker window:", 0.01, 1.0, 0.35, 0.05
        )
        self._walker_min_window_spin = self._add_double_spin(
            advanced_layout, "Min window:", 0.5, 20.0, 2.0, 0.5
        )
        self._walker_strength_spin = self._add_double_spin(
            advanced_layout, "Walker strength:", 0.01, 2.0, 0.5, 0.05
        )
        self._min_cuts_spin = self._add_int_spin(
            advanced_layout, "Min cuts:", 0, 64, 4
        )
        self._fallback_segments_spin = self._add_int_spin(
            advanced_layout, "Fallback segments:", 8, 256, 64
        )
        self._max_step_ratio_spin = self._add_double_spin(
            advanced_layout, "Max step ratio:", 1.0, 5.0, 1.8, 0.1
        )

        layout.addWidget(self._advanced_group)
        layout.addStretch()

        return self._widget

    def _add_int_spin(
        self, layout: QVBoxLayout, label: str, min_val: int, max_val: int, default: int
    ) -> QSpinBox:
        row = QHBoxLayout()
        name = QLabel(label)
        name.setFixedWidth(100)
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.valueChanged.connect(self._on_option_changed)
        row.addWidget(name)
        row.addWidget(spin, 1)
        layout.addLayout(row)
        return spin

    def _add_double_spin(
        self,
        layout: QVBoxLayout,
        label: str,
        min_val: float,
        max_val: float,
        default: float,
        step: float,
    ) -> QDoubleSpinBox:
        row = QHBoxLayout()
        name = QLabel(label)
        name.setFixedWidth(100)
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setValue(default)
        spin.valueChanged.connect(self._on_option_changed)
        row.addWidget(name)
        row.addWidget(spin, 1)
        layout.addLayout(row)
        return spin

    def _is_snapping_enabled(self) -> bool:
        return self._snapping_check.isChecked() if self._snapping_check else True

    def _update_grid_dependent_controls(self) -> None:
        snapping = self._is_snapping_enabled()
        if self._pixel_size_spin:
            self._pixel_size_spin.setEnabled(snapping)
        if self._output_mode_combo:
            self._output_mode_combo.setEnabled(snapping)
        if self._upscale_spin:
            self._upscale_spin.setEnabled(snapping and self._is_downsample_mode())
        if self._advanced_group:
            self._advanced_group.setEnabled(snapping)

    def _on_snapping_changed(self, _state: int) -> None:
        self._update_grid_dependent_controls()
        self._notify_change()

    def _on_output_mode_changed(self, _index: int):
        self._update_grid_dependent_controls()
        self._notify_change()

    def _is_downsample_mode(self) -> bool:
        if not self._output_mode_combo:
            return True
        return self._output_mode_combo.currentText() == self.OUTPUT_DOWNSAMPLE

    def _on_option_changed(self, *args):
        self._notify_change()

    def _notify_change(self):
        if self._callback:
            self._callback()

    def _build_config(self) -> SnapperConfig:
        pixel_size = self._pixel_size_spin.value() if self._pixel_size_spin else 0
        pixel_override = None if pixel_size == 0 else float(pixel_size)

        config = SnapperConfig(
            quantize_before_detect=False,
            enable_snapping=self._is_snapping_enabled(),
            pixel_size_override=pixel_override,
        )

        if self._advanced_group and self._advanced_group.isChecked():
            config = replace(
                config,
                peak_threshold_multiplier=self._peak_threshold_spin.value(),
                peak_distance_filter=self._peak_distance_spin.value(),
                walker_search_window_ratio=self._walker_window_spin.value(),
                walker_min_search_window=self._walker_min_window_spin.value(),
                walker_strength_threshold=self._walker_strength_spin.value(),
                min_cuts_per_axis=self._min_cuts_spin.value(),
                fallback_target_segments=self._fallback_segments_spin.value(),
                max_step_ratio=self._max_step_ratio_spin.value(),
            )

        return config

    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return None

        if self._pixel_size_spin is None:
            return image

        try:
            config = self._build_config()
            downsample = self._is_downsample_mode()
            output_scale = self._upscale_spin.value() if self._upscale_spin else 1
            if not downsample:
                output_scale = 1
            return process_pixel_snap(image, config, downsample, output_scale)
        except Exception as e:
            print(f"Pixel Snapper error: {e}")
            return image.copy()

    def set_source_dimensions(self, width: int, height: int) -> None:
        self._source_width = width
        self._source_height = height
        if self._pixel_size_spin and width > 0 and height > 0:
            max_override = max(1, min(width, height) // 2)
            self._pixel_size_spin.setMaximum(max_override)

    def on_parameters_changed(self, callback: Callable[[], None]) -> None:
        self._callback = callback

    def save_settings(self, config: ConfigManager) -> None:
        self._save_check(config, "enable_snapping", self._snapping_check)
        self._save_spin(config, "pixel_size", self._pixel_size_spin)
        self._save_combo(config, "output_mode", self._output_mode_combo)
        self._save_spin(config, "upscale", self._upscale_spin)
        self._save_check(config, "advanced_enabled", self._advanced_group, is_group=True)
        self._save_double(config, "peak_threshold", self._peak_threshold_spin)
        self._save_spin(config, "peak_distance", self._peak_distance_spin)
        self._save_double(config, "walker_window", self._walker_window_spin)
        self._save_double(config, "walker_min_window", self._walker_min_window_spin)
        self._save_double(config, "walker_strength", self._walker_strength_spin)
        self._save_spin(config, "min_cuts", self._min_cuts_spin)
        self._save_spin(config, "fallback_segments", self._fallback_segments_spin)
        self._save_double(config, "max_step_ratio", self._max_step_ratio_spin)

    def load_settings(self, config: ConfigManager) -> None:
        self.get_widget()

        self._load_check(config, "enable_snapping", self._snapping_check, True)
        self._load_spin(config, "pixel_size", self._pixel_size_spin, 0)
        self._load_combo(config, "output_mode", self._output_mode_combo, self.OUTPUT_DOWNSAMPLE)
        self._load_spin(config, "upscale", self._upscale_spin, 1)

        advanced = config.load_operator_setting(self.name, "advanced_enabled", False)
        if self._advanced_group:
            if isinstance(advanced, str):
                advanced = advanced.lower() == "true"
            self._advanced_group.setChecked(bool(advanced))

        self._load_double(config, "peak_threshold", self._peak_threshold_spin, 0.2)
        self._load_spin(config, "peak_distance", self._peak_distance_spin, 4)
        self._load_double(config, "walker_window", self._walker_window_spin, 0.35)
        self._load_double(config, "walker_min_window", self._walker_min_window_spin, 2.0)
        self._load_double(config, "walker_strength", self._walker_strength_spin, 0.5)
        self._load_spin(config, "min_cuts", self._min_cuts_spin, 4)
        self._load_spin(config, "fallback_segments", self._fallback_segments_spin, 64)
        self._load_double(config, "max_step_ratio", self._max_step_ratio_spin, 1.8)

        self._on_snapping_changed(
            Qt.CheckState.Checked.value
            if self._snapping_check and self._snapping_check.isChecked()
            else Qt.CheckState.Unchecked.value
        )
        self._on_output_mode_changed(self._output_mode_combo.currentIndex())

    def reset_to_defaults(self) -> None:
        if self._snapping_check:
            self._snapping_check.setChecked(True)
        if self._pixel_size_spin:
            self._pixel_size_spin.setValue(0)
        if self._output_mode_combo:
            self._output_mode_combo.setCurrentText(self.OUTPUT_DOWNSAMPLE)
        if self._upscale_spin:
            self._upscale_spin.setValue(1)
        if self._advanced_group:
            self._advanced_group.setChecked(False)
        if self._peak_threshold_spin:
            self._peak_threshold_spin.setValue(0.2)
        if self._peak_distance_spin:
            self._peak_distance_spin.setValue(4)
        if self._walker_window_spin:
            self._walker_window_spin.setValue(0.35)
        if self._walker_min_window_spin:
            self._walker_min_window_spin.setValue(2.0)
        if self._walker_strength_spin:
            self._walker_strength_spin.setValue(0.5)
        if self._min_cuts_spin:
            self._min_cuts_spin.setValue(4)
        if self._fallback_segments_spin:
            self._fallback_segments_spin.setValue(64)
        if self._max_step_ratio_spin:
            self._max_step_ratio_spin.setValue(1.8)
        self._update_grid_dependent_controls()

    def _save_spin(self, config: ConfigManager, key: str, spin: Optional[QSpinBox]) -> None:
        if spin:
            config.save_operator_setting(self.name, key, spin.value())

    def _save_double(self, config: ConfigManager, key: str, spin: Optional[QDoubleSpinBox]) -> None:
        if spin:
            config.save_operator_setting(self.name, key, spin.value())

    def _save_check(
        self,
        config: ConfigManager,
        key: str,
        widget: Optional[QCheckBox],
        is_group: bool = False,
    ) -> None:
        if widget:
            if is_group and isinstance(widget, QGroupBox):
                config.save_operator_setting(self.name, key, widget.isChecked())
            elif isinstance(widget, QCheckBox):
                config.save_operator_setting(self.name, key, widget.isChecked())

    def _save_combo(self, config: ConfigManager, key: str, combo: Optional[QComboBox]) -> None:
        if combo:
            config.save_operator_setting(self.name, key, combo.currentText())

    def _load_spin(
        self, config: ConfigManager, key: str, spin: Optional[QSpinBox], default: int
    ) -> None:
        if not spin:
            return
        value = config.load_operator_setting(self.name, key, default)
        if isinstance(value, str):
            value = int(value)
        spin.setValue(int(value))

    def _load_double(
        self, config: ConfigManager, key: str, spin: Optional[QDoubleSpinBox], default: float
    ) -> None:
        if not spin:
            return
        value = config.load_operator_setting(self.name, key, default)
        if isinstance(value, str):
            value = float(value)
        spin.setValue(float(value))

    def _load_check(
        self, config: ConfigManager, key: str, check: Optional[QCheckBox], default: bool
    ) -> None:
        if not check:
            return
        value = config.load_operator_setting(self.name, key, default)
        if isinstance(value, str):
            value = value.lower() == "true"
        check.setChecked(bool(value))

    def _load_combo(
        self, config: ConfigManager, key: str, combo: Optional[QComboBox], default: str
    ) -> None:
        if not combo:
            return
        value = config.load_operator_setting(self.name, key, default)
        index = combo.findText(str(value))
        if index >= 0:
            combo.setCurrentIndex(index)
