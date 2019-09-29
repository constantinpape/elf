[![Build Status](https://travis-ci.com/constantinpape/elf.svg?branch=master)](https://travis-ci.com/constantinpape/elf)

# elf

This repository implements common functionality for (large-scale) bio-medical image analysis:
- **evaluation**: evaluation of partitions via rand index and variation of information
- **io**: common interface for different libraries / formats 
- **segmentation**: graph-partition based segmentation
- **skeleton**: skeletonization
- **transformation**: helper functions for affine transformations
- **wrapper**: volume wrappers for on-the-fly transformations

and more.

It is used by several down-stream dependencies:
[cluster_tools](https://github.com/constantinpape/cluster_tools), [heimdall](https://github.com/constantinpape/heimdall),
[skeletor](https://github.com/constantinpape/skeletor), [paintera_tools](https://github.com/constantinpape/paintera_tools), [pybdv](https://github.com/constantinpape/pybdv) and [ilastik](https://github.com/ilastik/ilastik).


## Installation

Install the package from source via
```
python setup.py install
```
or via conda
```
conda install -c conda-forge -c cpape elf
```

## Functionality

