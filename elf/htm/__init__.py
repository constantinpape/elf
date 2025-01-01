"""Visualization of high-throughput / high-content microscopy data via napari.

Examples for using this functionality are in [example/htm](https://github.com/constantinpape/elf/tree/master/example/htm).
"""

from .parser import parse_simple_htm
from .visualization import view_plate, view_positional_images
