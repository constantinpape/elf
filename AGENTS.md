# Library overview

This library implements segmentation and other image analysis functionality in python.
The functionality is implemented in `elf` and its submodules.

## Current task

The library relies heavily on implementations from `nifty`, `vigra`, and `affogato`, which are C++ bindings with python libraries.
However, these libraries are difficult to install (e.g. not available on PyPI), limiting accessibility.

We have developed a new library `bioimage-cpp` to bundle the needed functionality and to make it more easily available.
Now, we want to migrate `elf` to use `bioimage-cpp`to eliminate `nifty`, `vigra`, and `affogato` dependencies.
In addition, we also want to use more efficient implementation from `bioimage-cpp` over `scipy`, `skimage` where available.
Please refer to `bioimage-cpp`'s migration guide for how to migrate the functionality:
https://raw.githubusercontent.com/computational-cell-analytics/bioimage-cpp/refs/heads/main/MIGRATION_GUIDE.md

We will go through the sub-modules one by one to migrate functionality. For each sub-module please follow this strategy:
- First check for sufficient test coverage and add further tests to increase it.
- Then check for general issues or missing functionality (indicated by TODOs etc.) in the sub-module and fix them.
- Then check for functionality to migrate.
    - If you cannot find a matching function in `bioimage-cpp` let me know and it will be implemented there.
- Then do the migration, run tests to ensure its success.

We will go through each of these steps with planning mode.

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
