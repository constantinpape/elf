[![Build Status](https://github.com/constantinpape/elf/workflows/build/badge.svg)](https://github.com/constantinpape/elf/actions)

# elf

This repository implements common functionality for (large-scale) bio-medical image analysis:
- **evaluation**: evaluation of partitions via rand index and variation of information
- **io**: common interface for different libraries / formats 
- **parallel**: parallel / larger than memory implementation of common numpy functions
- **segmentation**: graph-partition based segmentation
- **skeleton**: skeletonization
- **transformation**: helper functions for affine transformations
- **wrapper**: volume wrappers for on-the-fly transformations

and more.

See `examples` for some usage examples. For processing large data on a cluster, check out [cluster_tools](https://github.com/constantinpape/cluster_tools), which uses a lot of `elf` functionality internally.

It is used by several down-stream dependencies:
- [cluster_tools](https://github.com/constantinpape/cluster_tools)
- [paintera_tools](https://github.com/constantinpape/paintera_tools)
- [pybdv](https://github.com/constantinpape/pybdv)
- [ilastik](https://github.com/ilastik/ilastik)
- [mobie-python](https://github.com/mobie/mobie-utils-python).

## Installation

Install the package from source via
```
python setup.py install
```
or via conda
```
conda install -c conda-forge python-elf
```
