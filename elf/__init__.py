"""[elf](https://github.com/constantinpape/elf) implements image analysis functionality for large microscopy data.

# Overview

`elf` provides functionality for different image analysis tasks. The main functionality is:
- `elf.evaluation`: Common metrics for evaluating segmentation results.
- `elf.io`: Common interface for reading file formats for large microscopy data.
- `elf.parallel`: Parallel implementations of image analysis functions.
- `elf.segmentation`: Segmentation functions based on clustering, (lifted) multicut, mutex watershed and more.
- `elf.tracking`: Graph-based tracking algorithms.
- `elf.wrapper`: Wrapper for large microscopy data to enable on-the-fly processing.

# Installation

`elf` is available on conda-forge. You can install it into an existing conda environment via:
```
conda install -c conda-forge python-elf
```

We also provide a environment for a development environment. To set it up:
1. Clone the elf repository:
```
git clone https://github.com/constantinpape/elf
```
2. Enter the root elf directory:
```
cd elf
```
3. Create the development environment:
```
conda create -f environment.yaml
```
4. Activate the environment:
```
conda activate elf-dev
```
5. Install elf in development mode:
```
pip install -e .
```

# Usage & Examples

Example scripts for many of `elf`'s features can be found in [example](https://github.com/constantinpape/elf/tree/master/example).

`elf` also provides command line functionality. Currently provided are the command line interfaces:
- `view_container`: Visualize the content of any file supported by `elf.io` with napari.
"""

from .__version__ import __version__
