"""
Dither Operator - Reduce palette with dithering to smooth color banding
"""
from typing import Callable, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QGroupBox, QSlider, QComboBox
)

from core.config_manager import ConfigManager
from core.dither_algo import (
    DITHER_ATKINSON,
    DITHER_FLOYD_STEINBERG,
    DITHER_NONE,
    DITHER_ORDERED,
    dither_to_palette,
)
from core.operator_base import OperatorBase
from core.pixel_snapper_algo import (
    COLOR_SPACE_HSV,
    COLOR_SPACE_LAB,
    COLOR_SPACE_RGB,
    SnapperConfig,
    compute_palette,
)


class DitherOperator(OperatorBase):
    """Reduce colors with dithering to smooth banding after quantization."""

    CHANNEL_LABELS = {
        COLOR_SPACE_RGB: ("R:", "G:", "B:"),
        COLOR_SPACE_LAB: ("L:", "a:", "b:"),
        COLOR_SPACE_HSV: ("H:", "S:", "V:"),
    }

    def __init__(self):
        self._widget: Optional[QWidget] = None
        self._callback: Optional[Callable] = None

        self._color_space_combo: Optional[QComboBox] = None
        self._num_colors_spin: Optional[QSpinBox] = None
        self._iterations_spin: Optional[QSpinBox] = None
        self._dither_combo: Optional[QComboBox] = None
        self._weight_sliders: Tuple[Optional[QSlider], Optional[QSlider], Optional[QSlider]] = (
            None, None, None
        )
        self._weight_labels: Tuple[Optional[QLabel], Optional[QLabel], Optional[QLabel]] = (
            None, None, None
        )
        self._weight_value_labels: Tuple[Optional[QLabel], Optional[QLabel], Optional[QLabel]] = (
            None, None, None
        )

    @property
    def name(self) -> str:
        return "Dither"

    def get_widget(self) -> QWidget:
        if self._widget is not None:
            return self._widget

        self._widget = QWidget()
        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        main_group = QGroupBox("Palette Settings")
        main_layout = QVBoxLayout(main_group)
        main_layout.setSpacing(4)

        space_layout = QHBoxLayout()
        space_label = QLabel("Color space:")
        space_label.setFixedWidth(70)
        self._color_space_combo = QComboBox()
        self._color_space_combo.addItems([COLOR_SPACE_RGB, COLOR_SPACE_LAB, COLOR_SPACE_HSV])
        self._color_space_combo.currentIndexChanged.connect(self._on_colorspace_changed)
        space_layout.addWidget(space_label)
        space_layout.addWidget(self._color_space_combo, 1)
        main_layout.addLayout(space_layout)

        colors_layout = QHBoxLayout()
        colors_label = QLabel("Colors:")
        colors_label.setFixedWidth(70)
        self._num_colors_spin = QSpinBox()
        self._num_colors_spin.setRange(2, 256)
        self._num_colors_spin.setValue(16)
        self._num_colors_spin.valueChanged.connect(self._on_option_changed)
        colors_layout.addWidget(colors_label)
        colors_layout.addWidget(self._num_colors_spin, 1)
        main_layout.addLayout(colors_layout)

        iter_layout = QHBoxLayout()
        iter_label = QLabel("Iterations:")
        iter_label.setFixedWidth(70)
        self._iterations_spin = QSpinBox()
        self._iterations_spin.setRange(1, 50)
        self._iterations_spin.setValue(15)
        self._iterations_spin.valueChanged.connect(self._on_option_changed)
        iter_layout.addWidget(iter_label)
        iter_layout.addWidget(self._iterations_spin, 1)
        main_layout.addLayout(iter_layout)

        dither_layout = QHBoxLayout()
        dither_label = QLabel("Dithering:")
        dither_label.setFixedWidth(70)
        self._dither_combo = QComboBox()
        self._dither_combo.addItems([
            DITHER_NONE,
            DITHER_FLOYD_STEINBERG,
            DITHER_ATKINSON,
            DITHER_ORDERED,
        ])
        self._dither_combo.setCurrentText(DITHER_FLOYD_STEINBERG)
        self._dither_combo.setToolTip(
            "Error diffusion smooths banding between palette colors.\n"
            "Floyd-Steinberg: best general smoothing.\n"
            "Atkinson: softer, less noisy.\n"
            "Ordered: retro tiled pattern."
        )
        self._dither_combo.currentIndexChanged.connect(self._on_option_changed)
        dither_layout.addWidget(dither_label)
        dither_layout.addWidget(self._dither_combo, 1)
        main_layout.addLayout(dither_layout)

        layout.addWidget(main_group)

        weights_group = QGroupBox("Channel Weights")
        weights_layout = QVBoxLayout(weights_group)
        weights_desc = QLabel("Adjust importance of each channel during palette matching")
        weights_desc.setStyleSheet("color: #666; font-size: 10px;")
        weights_layout.addWidget(weights_desc)

        sliders: list = []
        labels: list = []
        value_labels: list = []
        for i in range(3):
            row = QHBoxLayout()
            name = QLabel("")
            name.setFixedWidth(70)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(33 if i < 2 else 34)
            slider.valueChanged.connect(lambda v, idx=i: self._on_weight_changed(idx, v))
            value_label = QLabel("33%")
            value_label.setFixedWidth(35)
            row.addWidget(name)
            row.addWidget(slider, 1)
            row.addWidget(value_label)
            weights_layout.addLayout(row)
            sliders.append(slider)
            labels.append(name)
            value_labels.append(value_label)

        self._weight_sliders = tuple(sliders)
        self._weight_labels = tuple(labels)
        self._weight_value_labels = tuple(value_labels)
        layout.addWidget(weights_group)
        layout.addStretch()

        self._update_channel_labels()
        return self._widget

    def _current_color_space(self) -> str:
        if not self._color_space_combo:
            return COLOR_SPACE_RGB
        return self._color_space_combo.currentText()

    def _current_dither_method(self) -> str:
        if not self._dither_combo:
            return DITHER_FLOYD_STEINBERG
        return self._dither_combo.currentText()

    def _update_channel_labels(self) -> None:
        labels = self.CHANNEL_LABELS.get(
            self._current_color_space(), self.CHANNEL_LABELS[COLOR_SPACE_RGB]
        )
        for i, text in enumerate(labels):
            if self._weight_labels[i]:
                self._weight_labels[i].setText(text)

    def _on_colorspace_changed(self, _index: int) -> None:
        self._update_channel_labels()
        self._notify_change()

    def _on_weight_changed(self, index: int, value: int) -> None:
        if self._weight_value_labels[index]:
            self._weight_value_labels[index].setText(f"{value}%")
        self._notify_change()

    def _get_channel_weights(self) -> Tuple[float, float, float]:
        values = []
        for slider in self._weight_sliders:
            values.append(float(slider.value()) if slider else 33.0)
        return (values[0], values[1], values[2])

    def _on_option_changed(self, *args):
        self._notify_change()

    def _notify_change(self):
        if self._callback:
            self._callback()

    def _build_config(self) -> SnapperConfig:
        return SnapperConfig(
            k_colors=self._num_colors_spin.value() if self._num_colors_spin else 16,
            max_kmeans_iterations=self._iterations_spin.value() if self._iterations_spin else 15,
            color_space=self._current_color_space(),
            channel_weights=self._get_channel_weights(),
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return None

        if self._num_colors_spin is None:
            return image

        try:
            config = self._build_config()
            palette = compute_palette(image, config)
            if len(palette) == 0:
                return image.copy()
            return dither_to_palette(
                image,
                palette,
                self._current_dither_method(),
                config.color_space,
                config.channel_weights,
            )
        except Exception as e:
            print(f"Dither error: {e}")
            return image.copy()

    def on_parameters_changed(self, callback: Callable[[], None]) -> None:
        self._callback = callback

    def save_settings(self, config: ConfigManager) -> None:
        if self._color_space_combo:
            config.save_operator_setting(self.name, "color_space", self._color_space_combo.currentText())
        if self._num_colors_spin:
            config.save_operator_setting(self.name, "num_colors", self._num_colors_spin.value())
        if self._iterations_spin:
            config.save_operator_setting(self.name, "iterations", self._iterations_spin.value())
        if self._dither_combo:
            config.save_operator_setting(self.name, "dither_method", self._dither_combo.currentText())
        for i, slider in enumerate(self._weight_sliders):
            if slider:
                config.save_operator_setting(self.name, f"weight_{i}", slider.value())

    def load_settings(self, config: ConfigManager) -> None:
        self.get_widget()

        color_space = config.load_operator_setting(self.name, "color_space", COLOR_SPACE_RGB)
        if self._color_space_combo:
            index = self._color_space_combo.findText(str(color_space))
            if index >= 0:
                self._color_space_combo.setCurrentIndex(index)

        num_colors = config.load_operator_setting(self.name, "num_colors", 16)
        if self._num_colors_spin:
            if isinstance(num_colors, str):
                num_colors = int(num_colors)
            self._num_colors_spin.setValue(int(num_colors))

        iterations = config.load_operator_setting(self.name, "iterations", 15)
        if self._iterations_spin:
            if isinstance(iterations, str):
                iterations = int(iterations)
            self._iterations_spin.setValue(int(iterations))

        dither_method = config.load_operator_setting(self.name, "dither_method", DITHER_FLOYD_STEINBERG)
        if self._dither_combo:
            index = self._dither_combo.findText(str(dither_method))
            if index >= 0:
                self._dither_combo.setCurrentIndex(index)

        defaults = (33, 33, 34)
        for i, slider in enumerate(self._weight_sliders):
            if not slider:
                continue
            value = config.load_operator_setting(self.name, f"weight_{i}", defaults[i])
            if isinstance(value, str):
                value = int(value)
            slider.setValue(int(value))
            if self._weight_value_labels[i]:
                self._weight_value_labels[i].setText(f"{value}%")

        self._update_channel_labels()

    def reset_to_defaults(self) -> None:
        if self._color_space_combo:
            self._color_space_combo.setCurrentText(COLOR_SPACE_RGB)
        if self._num_colors_spin:
            self._num_colors_spin.setValue(16)
        if self._iterations_spin:
            self._iterations_spin.setValue(15)
        if self._dither_combo:
            self._dither_combo.setCurrentText(DITHER_FLOYD_STEINBERG)
        defaults = (33, 33, 34)
        for i, slider in enumerate(self._weight_sliders):
            if slider:
                slider.setValue(defaults[i])
            if self._weight_value_labels[i]:
                self._weight_value_labels[i].setText(f"{defaults[i]}%")
        self._update_channel_labels()
