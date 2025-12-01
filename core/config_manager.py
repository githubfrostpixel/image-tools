"""
Configuration Manager - Handles persistent settings using QSettings
"""
from PyQt6.QtCore import QSettings, QByteArray
from typing import Any, Optional


class ConfigManager:
    """Manages application settings using QSettings"""
    
    def __init__(self):
        self.settings = QSettings("ImageTool", "ImageProcessingTool")
    
    # ============== Window State ==============
    
    def save_window_geometry(self, geometry: QByteArray) -> None:
        """Save main window geometry"""
        self.settings.setValue("window/geometry", geometry)
    
    def load_window_geometry(self) -> Optional[QByteArray]:
        """Load main window geometry"""
        value = self.settings.value("window/geometry")
        if isinstance(value, QByteArray):
            return value
        return None
    
    def save_splitter_state(self, state: QByteArray) -> None:
        """Save splitter widget state"""
        self.settings.setValue("window/splitter_state", state)
    
    def load_splitter_state(self) -> Optional[QByteArray]:
        """Load splitter widget state"""
        value = self.settings.value("window/splitter_state")
        if isinstance(value, QByteArray):
            return value
        return None
    
    # ============== File Paths ==============
    
    def save_last_input_dir(self, path: str) -> None:
        """Save last used input directory"""
        self.settings.setValue("paths/last_input_dir", path)
    
    def load_last_input_dir(self) -> str:
        """Load last used input directory"""
        return self.settings.value("paths/last_input_dir", "")
    
    def save_last_output_dir(self, path: str) -> None:
        """Save last used output directory"""
        self.settings.setValue("paths/last_output_dir", path)
    
    def load_last_output_dir(self) -> str:
        """Load last used output directory"""
        return self.settings.value("paths/last_output_dir", "")
    
    # ============== Export Settings ==============
    
    def save_export_format(self, format_ext: str) -> None:
        """Save last used export format"""
        self.settings.setValue("export/format", format_ext)
    
    def load_export_format(self) -> str:
        """Load last used export format"""
        return self.settings.value("export/format", "PNG")
    
    # ============== Operator Settings ==============
    
    def save_operator_setting(self, operator_name: str, key: str, value: Any) -> None:
        """Save a setting for a specific operator"""
        full_key = f"operators/{operator_name}/{key}"
        self.settings.setValue(full_key, value)
    
    def load_operator_setting(self, operator_name: str, key: str, default: Any = None) -> Any:
        """Load a setting for a specific operator"""
        full_key = f"operators/{operator_name}/{key}"
        return self.settings.value(full_key, default)
    
    def save_last_operator(self, operator_name: str) -> None:
        """Save the last active operator tab"""
        self.settings.setValue("operators/last_active", operator_name)
    
    def load_last_operator(self) -> str:
        """Load the last active operator tab"""
        return self.settings.value("operators/last_active", "")
    
    # ============== Utility ==============
    
    def sync(self) -> None:
        """Force sync settings to storage"""
        self.settings.sync()

