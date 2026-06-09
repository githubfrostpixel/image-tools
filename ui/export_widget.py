"""
Export Widget - Format selection and save functionality
"""
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QLabel, QFileDialog
)
from PyQt6.QtCore import pyqtSignal

from core.image_processor import ImageProcessor


class ExportWidget(QWidget):
    """
    Widget for exporting processed images.
    
    Features:
    - Format selection dropdown
    - Save button that opens file dialog
    
    Signals:
        export_requested: Emitted when user wants to export (file_path, format)
    """
    
    export_requested = pyqtSignal(str, str)
    
    # Available export formats
    FORMATS = ["PNG", "JPG", "BMP", "TIFF", "WEBP"]
    
    def __init__(self, parent=None, compact: bool = False):
        super().__init__(parent)
        self._compact = compact
        self._last_directory = ""
        self._source_filename = ""
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the UI"""
        if self._compact:
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
        else:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
        
        format_label = QLabel("Format:")
        format_label.setFixedWidth(50)
        
        self._format_combo = QComboBox()
        self._format_combo.addItems(self.FORMATS)
        self._format_combo.setCurrentText("PNG")
        if self._compact:
            self._format_combo.setFixedWidth(80)
        
        self._export_btn = QPushButton("Save" if self._compact else "Export Image")
        self._export_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #666;
            }
        """)
        self._export_btn.clicked.connect(self._on_export_clicked)
        self._export_btn.setEnabled(False)
        
        if self._compact:
            layout.addWidget(format_label)
            layout.addWidget(self._format_combo)
            layout.addWidget(self._export_btn)
        else:
            format_layout = QHBoxLayout()
            format_layout.setSpacing(4)
            format_layout.addWidget(format_label)
            format_layout.addWidget(self._format_combo, 1)
            layout.addLayout(format_layout)
            layout.addWidget(self._export_btn)
    
    def _on_export_clicked(self):
        """Handle export button click"""
        # Generate default filename
        default_name = "processed_image"
        if self._source_filename:
            source_path = Path(self._source_filename)
            default_name = f"{source_path.stem}_processed"
        
        format_ext = self._format_combo.currentText().lower()
        default_name = f"{default_name}.{format_ext}"
        
        # Build filter string
        format_filter = self._get_format_filter()
        
        # Open save dialog
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            str(Path(self._last_directory) / default_name) if self._last_directory else default_name,
            format_filter
        )
        
        if file_path:
            # Update last directory
            self._last_directory = str(Path(file_path).parent)
            
            # Ensure correct extension
            path = Path(file_path)
            expected_ext = f".{format_ext}"
            if path.suffix.lower() != expected_ext:
                file_path = str(path.with_suffix(expected_ext))
            
            self.export_requested.emit(file_path, format_ext)
    
    def _get_format_filter(self) -> str:
        """Get file filter for save dialog"""
        format_ext = self._format_combo.currentText()
        filters = {
            "PNG": "PNG Files (*.png)",
            "JPG": "JPEG Files (*.jpg *.jpeg)",
            "BMP": "BMP Files (*.bmp)",
            "TIFF": "TIFF Files (*.tiff *.tif)",
            "WEBP": "WebP Files (*.webp)"
        }
        return filters.get(format_ext, "All Files (*.*)")
    
    def set_enabled(self, enabled: bool):
        """Enable or disable export functionality"""
        self._export_btn.setEnabled(enabled)
    
    def set_source_filename(self, filename: str):
        """Set the source filename for default export name"""
        self._source_filename = filename
    
    def set_last_directory(self, directory: str):
        """Set the last used directory"""
        self._last_directory = directory
    
    def get_last_directory(self) -> str:
        """Get the last used directory"""
        return self._last_directory
    
    def get_format(self) -> str:
        """Get currently selected format"""
        return self._format_combo.currentText()
    
    def set_format(self, format_ext: str):
        """Set the selected format"""
        format_ext = format_ext.upper()
        index = self._format_combo.findText(format_ext)
        if index >= 0:
            self._format_combo.setCurrentIndex(index)

