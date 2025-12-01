"""
Image Viewer - Zoomable, pannable image display with transparency support
"""
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QSlider, QHBoxLayout, QLabel
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QBrush, QColor, QWheelEvent
from PyQt6.QtCore import Qt, QRectF

from core.image_processor import ImageProcessor


class CheckerboardBackground(QGraphicsPixmapItem):
    """Checkerboard pattern for transparency visualization"""
    
    @staticmethod
    def create_checkerboard(width: int, height: int, cell_size: int = 10) -> QPixmap:
        """Create a checkerboard pattern pixmap"""
        pixmap = QPixmap(width, height)
        painter = QPainter(pixmap)
        
        light = QColor(255, 255, 255)
        dark = QColor(200, 200, 200)
        
        for y in range(0, height, cell_size):
            for x in range(0, width, cell_size):
                color = light if ((x // cell_size) + (y // cell_size)) % 2 == 0 else dark
                painter.fillRect(x, y, cell_size, cell_size, color)
        
        painter.end()
        return pixmap


class ZoomableGraphicsView(QGraphicsView):
    """Graphics view with mouse wheel zoom support"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self._zoom_factor = 1.0
        self._zoom_callback = None
    
    def set_zoom_callback(self, callback):
        """Set callback to be called when zoom changes"""
        self._zoom_callback = callback
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming"""
        if event.angleDelta().y() > 0:
            factor = 1.15
        else:
            factor = 1 / 1.15
        
        new_zoom = self._zoom_factor * factor
        # Clamp zoom between 10% and 500%
        new_zoom = max(0.1, min(5.0, new_zoom))
        
        if new_zoom != self._zoom_factor:
            scale_factor = new_zoom / self._zoom_factor
            self._zoom_factor = new_zoom
            self.scale(scale_factor, scale_factor)
            
            if self._zoom_callback:
                self._zoom_callback(int(self._zoom_factor * 100))
    
    def set_zoom(self, zoom_percent: int):
        """Set zoom level from percentage"""
        new_zoom = zoom_percent / 100.0
        new_zoom = max(0.1, min(5.0, new_zoom))
        
        if new_zoom != self._zoom_factor:
            self.resetTransform()
            self._zoom_factor = new_zoom
            self.scale(new_zoom, new_zoom)
    
    def get_zoom(self) -> int:
        """Get current zoom percentage"""
        return int(self._zoom_factor * 100)
    
    def fit_in_view_auto(self, rect: QRectF):
        """Fit content in view while maintaining aspect ratio"""
        self.resetTransform()
        self._zoom_factor = 1.0
        
        if rect.isEmpty():
            return
        
        # Calculate scale to fit
        view_rect = self.viewport().rect()
        scale_x = view_rect.width() / rect.width()
        scale_y = view_rect.height() / rect.height()
        scale = min(scale_x, scale_y, 1.0)  # Don't scale up beyond 100%
        
        self._zoom_factor = scale
        self.scale(scale, scale)
        self.centerOn(rect.center())
        
        if self._zoom_callback:
            self._zoom_callback(int(self._zoom_factor * 100))


class ImageViewer(QWidget):
    """
    Image viewer widget with zoom, pan, and transparency support.
    
    Features:
    - Checkerboard background for transparency
    - Mouse wheel zoom
    - Zoom slider
    - Pan by dragging
    - Auto-fit on load
    """
    
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._title = title
        self._current_image = None
        self._checkerboard_item = None
        self._image_item = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # Title label
        if self._title:
            title_label = QLabel(self._title)
            title_label.setStyleSheet("font-weight: bold; padding: 2px;")
            layout.addWidget(title_label)
        
        # Graphics view
        self._scene = QGraphicsScene()
        self._view = ZoomableGraphicsView()
        self._view.setScene(self._scene)
        self._view.setMinimumSize(200, 150)
        self._view.set_zoom_callback(self._on_zoom_changed)
        layout.addWidget(self._view, 1)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.setContentsMargins(0, 0, 0, 0)
        
        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(45)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setRange(10, 500)
        self._zoom_slider.setValue(100)
        self._zoom_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self._zoom_slider.valueChanged.connect(self._on_slider_changed)
        
        fit_label = QLabel("Fit")
        fit_label.setStyleSheet("color: #666; font-size: 10px;")
        fit_label.setCursor(Qt.CursorShape.PointingHandCursor)
        fit_label.mousePressEvent = lambda e: self.fit_to_view()
        
        zoom_layout.addWidget(self._zoom_label)
        zoom_layout.addWidget(self._zoom_slider, 1)
        zoom_layout.addWidget(fit_label)
        
        layout.addLayout(zoom_layout)
    
    def _on_zoom_changed(self, zoom_percent: int):
        """Called when zoom changes from mouse wheel"""
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(zoom_percent)
        self._zoom_slider.blockSignals(False)
        self._zoom_label.setText(f"{zoom_percent}%")
    
    def _on_slider_changed(self, value: int):
        """Called when zoom slider changes"""
        self._view.set_zoom(value)
        self._zoom_label.setText(f"{value}%")
    
    def set_image(self, image: np.ndarray, auto_fit: bool = True):
        """
        Set the image to display.
        
        Args:
            image: Image in BGRA format
            auto_fit: Whether to auto-fit the image to view
        """
        self._current_image = image
        self._scene.clear()
        self._checkerboard_item = None
        self._image_item = None
        
        if image is None:
            return
        
        height, width = image.shape[:2]
        
        # Create checkerboard background
        checkerboard = CheckerboardBackground.create_checkerboard(width, height)
        self._checkerboard_item = self._scene.addPixmap(checkerboard)
        
        # Convert image to QPixmap
        rgba_image = ImageProcessor.bgra_to_rgba(image)
        
        if len(rgba_image.shape) == 3 and rgba_image.shape[2] == 4:
            qimage = QImage(
                rgba_image.data,
                width, height,
                rgba_image.strides[0],
                QImage.Format.Format_RGBA8888
            )
        else:
            qimage = QImage(
                rgba_image.data,
                width, height,
                rgba_image.strides[0],
                QImage.Format.Format_RGB888
            )
        
        pixmap = QPixmap.fromImage(qimage)
        self._image_item = self._scene.addPixmap(pixmap)
        
        # Set scene rect
        self._scene.setSceneRect(0, 0, width, height)
        
        if auto_fit:
            self.fit_to_view()
    
    def fit_to_view(self):
        """Fit the image to the view area"""
        if self._scene.sceneRect().isEmpty():
            return
        self._view.fit_in_view_auto(self._scene.sceneRect())
    
    def clear(self):
        """Clear the viewer"""
        self._current_image = None
        self._scene.clear()
        self._checkerboard_item = None
        self._image_item = None
    
    def get_zoom(self) -> int:
        """Get current zoom percentage"""
        return self._view.get_zoom()
    
    def set_zoom(self, zoom_percent: int):
        """Set zoom percentage"""
        self._zoom_slider.setValue(zoom_percent)

