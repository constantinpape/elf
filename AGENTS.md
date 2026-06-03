# Library overview

This library implements segmentation and other image analysis functionality in python.
The functionality is implemented in `elf` and its submodules.

## Library structure

All functionality lives in the `elf` package. Each submodule is self-contained and exposes its public API via its `__init__.py`. The `test` folder mirrors this layout, with one subfolder per submodule.

Top-level utilities:
- `elf/util.py`: General-purpose helpers used across submodules.
- `elf/__version__.py`: Single source of truth for the package version.

Submodules:
- `elf/io`: Unified interface for reading/writing large microscopy data. `files.py` is the main entry point (`open_file`); the `*_wrapper.py` files adapt specific backends (zarr, n5, mrc, knossos, nifti, intern, image stacks, etc.).
- `elf/wrapper`: Lazy array wrappers that apply transformations on-the-fly (affine, resize, caching). `base.py` defines the wrapper base class; `generic.py` provides a generic apply-function wrapper.
- `elf/parallel`: Blockwise/parallel implementations of array operations (label, relabel, watershed, distance transform, filters, size filter, unique, stats, copy). Use these for data too large to fit in memory.
- `elf/segmentation`: Core segmentation algorithms — (lifted) multicut, mutex watershed, GASP, clustering, watershed, plus blockwise variants (`blockwise_*_impl.py`), feature computation (`features.py`), embeddings, learning, stitching, postprocessing, and high-level pipelines in `workflows.py`.
- `elf/evaluation`: Segmentation metrics — rand index, variation of information, dice, cremi score, and matching-based scores.
- `elf/transformation`: Coordinate/image transformations, including affine transforms, resizing, NGFF transforms, and elastix/transformix wrappers.
- `elf/tracking`: Graph-based tracking (motile), with MaMuT import/export and shared utilities.
- `elf/mesh`: Mesh generation from segmentations, mesh I/O, and mesh-to-segmentation conversion.
- `elf/skeleton`: Skeletonization (`skeletonize.py`, `thinning.py`) and skeleton I/O.
- `elf/label_multiset`: Paintera-style label multiset data structure — creation, serialization, and the core data structure.
- `elf/htm`: High-throughput microscopy helpers — parsing and visualization.
- `elf/ilastik`: Interop with ilastik (e.g. carving).
- `elf/visualisation`: Visualization helpers (edges, grids, metrics, object/size views).
- `elf/color`: Color palettes for visualization.

Note the British spelling in `elf/visualisation`. Heavy numerical routines are increasingly backed by [bioimage-cpp](https://github.com/computational-cell-analytics/bioimage-cpp) (replacing the older affogato/nifty/vigra dependencies); some functions degrade gracefully or raise if an optional backend is missing.

## Coding standards

The code should be PEP8 compliant with 120 char line length limit (see below), doc strings use google conventions.
Tests are written with `unittest` and are located in the `test` folder.

## Installation, linting and testing

To install the library:
```bash
pip install -e .
```

To run the linter:
```bash
pyflakes <path/to/file.py>
flake8 <path/to/file.py> --max-line-length=120
```

To run all tests:
```bash
python -m unittest discover -s test -v
```

To run a specific test suite:
```bash
python test/<test-file.py>
```
with further modifiers to run only a selected test.
