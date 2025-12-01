"""
Operator Base - Abstract base class for all image operators
"""
from abc import ABC, abstractmethod
from typing import Callable, Optional
import numpy as np
from PyQt6.QtWidgets import QWidget

from .config_manager import ConfigManager


class OperatorBase(ABC):
    """
    Abstract base class for all image processing operators.
    
    To create a new operator:
    1. Create a new file in the operators/ directory
    2. Create a class that inherits from OperatorBase
    3. Implement all abstract methods
    4. The operator will be auto-discovered by the registry
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Display name for this operator (shown in tab).
        Must be unique among all operators.
        """
        pass
    
    @abstractmethod
    def get_widget(self) -> QWidget:
        """
        Return the parameter UI widget for this operator.
        This widget will be displayed in the operator tab.
        """
        pass
    
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process the input image and return the result.
        
        Args:
            image: Input image in BGRA format
            
        Returns:
            Processed image in BGRA format
        """
        pass
    
    @abstractmethod
    def on_parameters_changed(self, callback: Callable[[], None]) -> None:
        """
        Register a callback to be called when any parameter changes.
        This enables live preview functionality.
        
        Args:
            callback: Function to call when parameters change
        """
        pass
    
    @abstractmethod
    def save_settings(self, config: ConfigManager) -> None:
        """
        Save operator-specific settings to config.
        Called when application closes.
        
        Args:
            config: ConfigManager instance
        """
        pass
    
    @abstractmethod
    def load_settings(self, config: ConfigManager) -> None:
        """
        Load operator-specific settings from config.
        Called when application starts.
        
        Args:
            config: ConfigManager instance
        """
        pass
    
    def set_source_dimensions(self, width: int, height: int) -> None:
        """
        Called when a new image is loaded to inform operator of source dimensions.
        Override this method if your operator needs to know the source image size.
        
        Args:
            width: Source image width
            height: Source image height
        """
        pass
    
    def reset_to_defaults(self) -> None:
        """
        Reset all parameters to default values.
        Override this method to implement reset functionality.
        """
        pass

