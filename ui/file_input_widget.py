"""
File Input Widget - Browse button with drag-and-drop support
"""
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

from core.image_processor import ImageProcessor


class FileInputWidget(QWidget):
    """
    Widget for selecting input images via browse button or drag-and-drop.
    
    Signals:
        file_selected: Emitted when a valid image file is selected
    """
    
    file_selected = pyqtSignal(str)
    
    def __init__(self, parent=None, compact: bool = False):
        super().__init__(parent)
        self._compact = compact
        self._last_directory = ""
        self._current_file = ""
        
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
        
        # Drop zone / browse button
        self._drop_zone = DropZoneButton(compact=self._compact)
        self._drop_zone.clicked.connect(self._browse_file)
        self._drop_zone.file_dropped.connect(self._on_file_dropped)
        layout.addWidget(self._drop_zone)
        
        # File name label
        self._file_label = QLabel("No file selected")
        self._file_label.setStyleSheet("color: #666; font-size: 10px;")
        self._file_label.setWordWrap(not self._compact)
        if self._compact:
            self._file_label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            layout.addWidget(self._file_label, 1)
        else:
            self._file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self._file_label)
    
    def _browse_file(self):
        """Open file browser dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            self._last_directory,
            ImageProcessor.get_file_filter()
        )
        
        if file_path:
            self._set_file(file_path)
    
    def _on_file_dropped(self, file_path: str):
        """Handle file dropped on widget"""
        self._set_file(file_path)
    
    def _set_file(self, file_path: str):
        """Set the selected file"""
        path = Path(file_path)
        
        if not path.exists():
            return
        
        if path.suffix.lower() not in ImageProcessor.SUPPORTED_FORMATS:
            self._file_label.setText("Unsupported format")
            return
        
        self._current_file = file_path
        self._last_directory = str(path.parent)
        
        # Show filename (truncate if too long)
        name = path.name
        if len(name) > 30:
            name = name[:27] + "..."
        self._file_label.setText(name)
        
        self.file_selected.emit(file_path)
    
    def set_last_directory(self, directory: str):
        """Set the last used directory"""
        self._last_directory = directory
    
    def get_last_directory(self) -> str:
        """Get the last used directory"""
        return self._last_directory
    
    def get_current_file(self) -> str:
        """Get the currently selected file path"""
        return self._current_file


class DropZoneButton(QPushButton):
    """
    Button that also accepts drag-and-drop files.
    """
    
    file_dropped = pyqtSignal(str)
    
    def __init__(self, parent=None, compact: bool = False):
        super().__init__(parent)
        self._compact = compact
        if compact:
            self.setText("Browse Image")
            self.setMinimumHeight(32)
            self.setMaximumHeight(32)
        else:
            self.setText("Browse Image\n\nDrag & Drop Here")
            self.setMinimumHeight(80)
        self.setAcceptDrops(True)
        self._reset_style()
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                if Path(file_path).suffix.lower() in ImageProcessor.SUPPORTED_FORMATS:
                    event.acceptProposedAction()
                    self._set_drag_style()
                    return
        event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave"""
        self._reset_style()
    
    def dropEvent(self, event: QDropEvent):
        """Handle file drop"""
        self._reset_style()
        
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                if Path(file_path).suffix.lower() in ImageProcessor.SUPPORTED_FORMATS:
                    event.acceptProposedAction()
                    self.file_dropped.emit(file_path)
                    return
        event.ignore()
    
    def _set_drag_style(self):
        """Apply drag-over highlight style"""
        if self._compact:
            self.setStyleSheet("""
                QPushButton {
                    border: 2px dashed #4CAF50;
                    border-radius: 4px;
                    background-color: #e8f5e9;
                    color: #555;
                    font-size: 11px;
                    padding: 4px 12px;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    border: 2px dashed #4CAF50;
                    border-radius: 8px;
                    background-color: #e8f5e9;
                    color: #555;
                    font-size: 12px;
                    padding: 10px;
                }
            """)
    
    def _reset_style(self):
        """Reset to default style"""
        if self._compact:
            self.setStyleSheet("""
                QPushButton {
                    border: 2px dashed #888;
                    border-radius: 4px;
                    background-color: #f5f5f5;
                    color: #555;
                    font-size: 11px;
                    padding: 4px 12px;
                }
                QPushButton:hover {
                    border-color: #555;
                    background-color: #eee;
                }
                QPushButton:pressed {
                    background-color: #ddd;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    border: 2px dashed #888;
                    border-radius: 8px;
                    background-color: #f5f5f5;
                    color: #555;
                    font-size: 12px;
                    padding: 10px;
                }
                QPushButton:hover {
                    border-color: #555;
                    background-color: #eee;
                }
                QPushButton:pressed {
                    background-color: #ddd;
                }
            """)

