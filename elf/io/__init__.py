"""Common interface for reading and writing microscopy data formats for large imaging data.
"""

from .files import open_file, supported_extensions
from .files import is_group, is_dataset, is_h5py, is_z5py, is_knossos
