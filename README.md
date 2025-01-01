[![Build Status](https://github.com/constantinpape/elf/workflows/build/badge.svg)](https://github.com/constantinpape/elf/actions)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/python-elf/badges/version.svg)](https://anaconda.org/conda-forge/python-elf)
[![Documentation - Documentation](https://img.shields.io/badge/Documentation-Documentation-2ea44f)](https://constantinpape.github.io/elf/elf.html)

# elf

This repository implements common functionality for biomedical image analysis:
- **evaluation**: evaluation of partitions via rand index and variation of information
- **io**: common interface for different libraries / formats 
- **parallel**: parallel / larger than memory implementation of common numpy functions
- **segmentation**: graph-partition based segmentation
- **skeleton**: skeletonization
- **transformation**: helper functions for affine transformations
- **wrapper**: volume wrappers for on-the-fly transformations
- **tracking**: graph based tracking algorithms

and more. See [the documentation](https://constantinpape.github.io/elf/elf.html) for how to use elf.

See `examples` for some usage examples. For processing large data on a cluster, check out [cluster_tools](https://github.com/constantinpape/cluster_tools), which uses a lot of `elf` functionality internally.

It is used by several down-stream dependencies:
- [cluster_tools](https://github.com/constantinpape/cluster_tools)
- [paintera_tools](https://github.com/constantinpape/paintera_tools)
- [pybdv](https://github.com/constantinpape/pybdv)
- [ilastik](https://github.com/ilastik/ilastik)
- [mobie-python](https://github.com/mobie/mobie-utils-python)
- [plantseg](https://github.com/hci-unihd/plant-seg)

## Installation

Install the package from source and in development mode via
```
pip install -e .
```
or via conda
```
conda install -c conda-forge python-elf
```

## Functionality overview

**Segmentation:** `elf` implements graph-based segmentation using the implementations of multict, lifted multicut and other graph partitioning approaches from [nifty](https://github.com/DerThorsten/nifty).
Check out [the examples](https://github.com/constantinpape/elf/tree/master/example/segmentation) to see how to use this functionality for segmenting your data.

**Tracking:** `elf` implements graph-based tracking using the implementations from [motile](https://github.com/funkelab/motile).
Checkout [the examples](https://github.com/constantinpape/elf/tree/master/example/tracking) to see how to use this functionality to track your data.

In order to use this functionality you will need to install `motile`. You can do this via
```
conda install -c conda-forge -c funkelab -c gurobi ilpy
```
and then
```
pip install motile
```
