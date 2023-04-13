import runpy
from setuptools import setup, find_packages

__version__ = runpy.run_path("elf/__version__.py")["__version__"]

requires = [
    "numpy",
    "imageio",
    "scikit-image",
    "scikit-learn",
    "skan"
]


# optional dependencies for setuptools
extras = {
    "hdf5": "h5py",
    "zarr": "zarr",
    "n5": "pyn5",
    "cloud": "intern"
}

# dependencies only available via conda,
# we still collect them here, because the conda recipe
# gets its requirements from setuptools.
conda_only = ["vigra", "nifty", "z5py"]

# collect all dependencies for conda
conda_exclude = [
    "zarr",  # we don't need zarr dependencies in conda, because we use z5py
    "pyn5"  # pyn5 is not available on conda (and not needed due to z5py)
]
conda_all = conda_only + [v for v in extras.values() if v not in conda_exclude]
extras["conda_all"] = conda_all

# NOTE in case we want to support different conda flavors at some point, we
# can add keys to 'extras', e.g. 'conda_no_hdf5' without h5py

setup(
    name="elf",
    packages=find_packages(exclude=["test"]),
    version=__version__,
    author="Constantin Pape",
    install_requires=requires,
    extras_require=extras,
    url="https://github.com/constantinpape/elf",
    license="MIT",
    entry_points={
        "console_scripts": [
            "view_container = elf.visualisation.view_container:main",
        ]
    },
)
