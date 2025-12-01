# Image Processing Tool

A modular image processing application built with PyQt6 and OpenCV.

## Features

- **Modular Operator System**: Easy to add new image processing operators
- **Live Preview**: Changes apply in real-time as you adjust parameters
- **Transparency Support**: Full alpha channel support with checkerboard background visualization
- **Advanced Image Viewer**: Zoom (mouse wheel + slider), pan, auto-fit
- **Drag & Drop**: Drop images directly onto the application
- **Configuration Persistence**: Remembers window size, last directories, and operator settings

## Current Operators

### Downscale
Resize images with advanced pixel art support.

**Output Size**
- Width and height controls
- Preserve aspect ratio option

**Interpolation Algorithms**
- Nearest: Fast, pixelated - good for pixel art
- Linear: Balanced speed and quality
- Cubic: Smoother than linear
- Area: Best for downscaling, reduces aliasing
- Lanczos: Highest quality, best for upscaling

**Pixel Art Options**
- Crop transparent borders: Remove empty space around sprites before processing
- Binarize alpha: Threshold alpha to pure 0/255, removes anti-aliasing from edges
- Corrode: Erode alpha channel to make sprites thinner (1-10 px iterations)
- Add outline: Add colored 1px outline around opaque areas (hex, name, or RGB)

**Output Padding**
- Pad to fixed size: Center image in a fixed canvas (useful for sprite sheets)

### Add Border
Add hugging borders around non-transparent pixels for pixel art.

**Border Style**
- Black border (inner): 1-pixel border hugging the sprite contour
- White border (outer): Additional outer border around the black one

**Offset**
- X/Y offset: Shift original image relative to borders (creates shadow/highlight effect)

**Colors**
- Inner color: Customizable inner border color (default: black)
- Outer color: Customizable outer border color (default: white)

## Supported Formats

**Input**: PNG, JPG, JPEG, BMP, TIFF, TIF, WebP, GIF, ICO, PPM, PGM, PBM

**Output**: PNG, JPG, BMP, TIFF, WebP

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Project Structure

```
image-downscale-tool/
|-- main.py                 # Application entry point
|-- requirements.txt        # Dependencies
|
|-- core/                   # Core functionality
|   |-- config_manager.py   # Settings persistence
|   |-- image_processor.py  # Image I/O with alpha support
|   |-- operator_base.py    # Abstract base for operators
|   |-- operator_registry.py # Auto-discovery of operators
|
|-- operators/              # Image processing operators
|   |-- downscale.py        # Resize/downscale operator
|   |-- add_border.py       # Hugging border operator
|
|-- ui/                     # User interface
|   |-- main_window.py      # Main application window
|   |-- image_viewer.py     # Zoomable image viewer
|   |-- file_input_widget.py # File browser/drag-drop
|   |-- export_widget.py    # Export controls
```

## Adding New Operators

1. Create a new file in `operators/` directory (e.g., `border.py`)
2. Create a class inheriting from `OperatorBase`
3. Implement required methods:
   - `name`: Display name for the tab
   - `get_widget()`: Parameter UI widget
   - `process(image)`: Image processing logic
   - `on_parameters_changed(callback)`: Live preview support
   - `save_settings(config)`: Persist settings
   - `load_settings(config)`: Restore settings

The operator will be automatically discovered and added to the application.

## Requirements

- Python 3.8+
- PyQt6 >= 6.4.0
- OpenCV-Python >= 4.8.0
- NumPy >= 1.24.0
