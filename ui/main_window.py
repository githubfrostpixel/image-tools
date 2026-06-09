"""
Main Window - Primary application window with 3-row layout
"""
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QTabWidget, QScrollArea, QPushButton, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon

from core.config_manager import ConfigManager
from core.image_processor import ImageProcessor
from core.operator_registry import OperatorRegistry
from core.operator_base import OperatorBase
from ui.file_input_widget import FileInputWidget
from ui.image_viewer import ImageViewer
from ui.export_widget import ExportWidget


class MainWindow(QMainWindow):
    """
    Main application window.
    
    Layout:
    - Row 1: Input image zone (left), format + save (right)
    - Row 2: Original viewer (left), Processed viewer (right)
    - Row 3: Operator function tabs
    """
    
    def __init__(self):
        super().__init__()
        
        self._config = ConfigManager()
        self._registry = OperatorRegistry()
        self._current_operator: OperatorBase = None
        self._original_image: np.ndarray = None
        self._processed_image: np.ndarray = None
        
        # Debounce timer for live preview
        self._process_timer = QTimer()
        self._process_timer.setSingleShot(True)
        self._process_timer.timeout.connect(self._do_process)
        
        self._setup_ui()
        self._load_settings()
        self._connect_operators()
    
    def _setup_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Image Processing Tool")
        self.setMinimumSize(800, 600)
        
        # Set window icon
        icon_path = Path(__file__).parent.parent / "favicon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        
        # ============== Row 1: Input + Export ==============
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(8)
        
        self._file_input = FileInputWidget(compact=True)
        self._file_input.file_selected.connect(self._on_file_selected)
        toolbar_layout.addWidget(self._file_input, 1)

        self._use_as_input_btn = QPushButton("Use as Input")
        self._use_as_input_btn.setToolTip("Copy the processed image into the original input")
        self._use_as_input_btn.clicked.connect(self._on_use_as_input)
        self._use_as_input_btn.setEnabled(False)
        toolbar_layout.addWidget(self._use_as_input_btn)
        
        self._export_widget = ExportWidget(compact=True)
        self._export_widget.export_requested.connect(self._on_export_requested)
        toolbar_layout.addWidget(self._export_widget)
        
        main_layout.addWidget(toolbar_widget)
        
        # ============== Row 2: Viewers (stretch to fill) ==============
        self._viewer_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self._original_viewer = ImageViewer("Original")
        self._viewer_splitter.addWidget(self._original_viewer)
        
        self._processed_viewer = ImageViewer("Processed")
        self._viewer_splitter.addWidget(self._processed_viewer)
        
        self._viewer_splitter.setSizes([400, 400])
        main_layout.addWidget(self._viewer_splitter, 1)
        
        # ============== Row 3: Operator tabs (fixed height, scrollable content) ==============
        self._operator_tabs = QTabWidget()
        self._operator_tabs.setFixedHeight(300)
        self._operator_tabs.currentChanged.connect(self._on_operator_changed)
        
        operators = self._registry.get_operators()
        for operator in operators:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(operator.get_widget())
            self._operator_tabs.addTab(scroll, operator.name)
        
        main_layout.addWidget(self._operator_tabs, 0)
        
        # Set current operator
        if operators:
            self._current_operator = operators[0]
    
    def _connect_operators(self):
        """Connect parameter change callbacks for all operators"""
        for operator in self._registry.get_operators():
            operator.on_parameters_changed(self._schedule_process)
    
    def _load_settings(self):
        """Load settings from config"""
        # Window geometry
        geometry = self._config.load_window_geometry()
        if geometry:
            self.restoreGeometry(geometry)
        
        # Viewer splitter state
        viewer_splitter_state = self._config.load_viewer_splitter_state()
        if viewer_splitter_state:
            self._viewer_splitter.restoreState(viewer_splitter_state)
        
        # File paths
        last_input_dir = self._config.load_last_input_dir()
        if last_input_dir:
            self._file_input.set_last_directory(last_input_dir)
        
        last_output_dir = self._config.load_last_output_dir()
        if last_output_dir:
            self._export_widget.set_last_directory(last_output_dir)
        
        # Export format
        export_format = self._config.load_export_format()
        if export_format:
            self._export_widget.set_format(export_format)
        
        # Operator settings
        for operator in self._registry.get_operators():
            operator.load_settings(self._config)
        
        # Last active operator
        last_operator = self._config.load_last_operator()
        if last_operator:
            for i in range(self._operator_tabs.count()):
                if self._operator_tabs.tabText(i) == last_operator:
                    self._operator_tabs.setCurrentIndex(i)
                    break
    
    def _save_settings(self):
        """Save settings to config"""
        # Window geometry
        self._config.save_window_geometry(self.saveGeometry())
        
        # Viewer splitter state
        self._config.save_viewer_splitter_state(self._viewer_splitter.saveState())
        
        # File paths
        self._config.save_last_input_dir(self._file_input.get_last_directory())
        self._config.save_last_output_dir(self._export_widget.get_last_directory())
        
        # Export format
        self._config.save_export_format(self._export_widget.get_format())
        
        # Operator settings
        for operator in self._registry.get_operators():
            operator.save_settings(self._config)
        
        # Last active operator
        if self._current_operator:
            self._config.save_last_operator(self._current_operator.name)
        
        self._config.sync()
    
    def _set_output_actions_enabled(self, enabled: bool):
        """Enable or disable actions that require a processed image"""
        self._export_widget.set_enabled(enabled)
        self._use_as_input_btn.setEnabled(enabled)

    def _on_use_as_input(self):
        """Copy processed image into original input for chaining operators"""
        if self._processed_image is None:
            return

        self._original_image = self._processed_image.copy()
        self._original_viewer.set_image(self._original_image)

        width, height = ImageProcessor.get_image_dimensions(self._original_image)
        for operator in self._registry.get_operators():
            operator.set_source_dimensions(width, height)

        self._schedule_process()

    def _on_file_selected(self, file_path: str):
        """Handle file selection"""
        image = ImageProcessor.load_image(file_path)
        
        if image is None:
            QMessageBox.warning(self, "Error", "Failed to load image.")
            return
        
        self._original_image = image
        self._original_viewer.set_image(image)
        
        # Update export widget with source filename
        self._export_widget.set_source_filename(file_path)
        
        # Notify operators of source dimensions
        width, height = ImageProcessor.get_image_dimensions(image)
        for operator in self._registry.get_operators():
            operator.set_source_dimensions(width, height)
        
        # Process with current operator
        self._schedule_process()
    
    def _on_operator_changed(self, index: int):
        """Handle operator tab change"""
        operators = self._registry.get_operators()
        if 0 <= index < len(operators):
            self._current_operator = operators[index]
            self._schedule_process()
    
    def _schedule_process(self):
        """Schedule image processing with debounce"""
        self._process_timer.start(50)  # 50ms debounce
    
    def _do_process(self):
        """Perform image processing"""
        if self._original_image is None or self._current_operator is None:
            self._processed_image = None
            self._processed_viewer.clear()
            self._set_output_actions_enabled(False)
            return
        
        try:
            self._processed_image = self._current_operator.process(self._original_image)
            
            if self._processed_image is not None:
                self._processed_viewer.set_image(self._processed_image)
                self._set_output_actions_enabled(True)
            else:
                self._processed_viewer.clear()
                self._set_output_actions_enabled(False)
                
        except Exception as e:
            print(f"Processing error: {e}")
            self._processed_viewer.clear()
            self._set_output_actions_enabled(False)
    
    def _on_export_requested(self, file_path: str, format_ext: str):
        """Handle export request"""
        if self._processed_image is None:
            QMessageBox.warning(self, "Error", "No processed image to export.")
            return
        
        success = ImageProcessor.save_image(self._processed_image, file_path, format_ext)
        
        if success:
            QMessageBox.information(self, "Success", f"Image exported to:\n{file_path}")
        else:
            QMessageBox.warning(self, "Error", "Failed to export image.")
    
    def closeEvent(self, event):
        """Handle window close"""
        self._save_settings()
        event.accept()
