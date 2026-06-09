"""
Operator Registry - Auto-discovers and manages operators
"""
import importlib
import pkgutil
from pathlib import Path
from typing import List, Dict, Optional
import sys

from .operator_base import OperatorBase


class OperatorRegistry:
    """
    Registry that auto-discovers and manages image processing operators.
    
    Scans the operators/ directory for classes that inherit from OperatorBase
    and makes them available for use in the application.
    """
    
    # Define operator display order
    OPERATOR_ORDER = [
        "Downscale",
        "Add Border",
        "Color Quantize",
        "Dither",
        "Pixel Snapper"
    ]
    
    def __init__(self):
        self._operators: Dict[str, OperatorBase] = {}
        self._discover_operators()
    
    def _discover_operators(self) -> None:
        """Scan operators directory and instantiate all operators"""
        # Get the operators package path
        try:
            import operators
            package_path = Path(operators.__file__).parent
        except ImportError:
            return
        
        # Iterate through all modules in the operators package
        for module_info in pkgutil.iter_modules([str(package_path)]):
            if module_info.name.startswith('_'):
                continue
            
            try:
                # Import the module
                module = importlib.import_module(f"operators.{module_info.name}")
                
                # Find all classes that inherit from OperatorBase
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    
                    # Check if it's a class that inherits from OperatorBase
                    if (isinstance(attr, type) and 
                        issubclass(attr, OperatorBase) and 
                        attr is not OperatorBase):
                        
                        # Instantiate and register the operator
                        try:
                            operator_instance = attr()
                            self._operators[operator_instance.name] = operator_instance
                        except Exception as e:
                            print(f"Failed to instantiate operator {attr_name}: {e}")
                            
            except Exception as e:
                print(f"Failed to import operator module {module_info.name}: {e}")
    
    def get_operators(self) -> List[OperatorBase]:
        """
        Get list of all registered operators, sorted by display order.
        
        Returns:
            List of operator instances in the specified order
        """
        operators = list(self._operators.values())
        
        # Sort operators by predefined order
        def sort_key(op: OperatorBase) -> int:
            try:
                return self.OPERATOR_ORDER.index(op.name)
            except ValueError:
                # Operators not in the order list go to the end
                return len(self.OPERATOR_ORDER)
        
        operators.sort(key=sort_key)
        return operators
    
    def get_operator(self, name: str) -> Optional[OperatorBase]:
        """
        Get a specific operator by name.
        
        Args:
            name: Operator name
            
        Returns:
            Operator instance or None if not found
        """
        return self._operators.get(name)
    
    def get_operator_names(self) -> List[str]:
        """
        Get list of all registered operator names.
        
        Returns:
            List of operator names
        """
        return list(self._operators.keys())
    
    def reload_operators(self) -> None:
        """
        Reload all operators (useful for development/hot-reload).
        """
        self._operators.clear()
        
        # Clear cached imports
        modules_to_remove = [
            name for name in sys.modules 
            if name.startswith('operators.') and not name.endswith('__init__')
        ]
        for module_name in modules_to_remove:
            del sys.modules[module_name]
        
        self._discover_operators()

