import runpy
from setuptools import setup, find_packages

__version__ = runpy.run_path("elf/__version__.py")["__version__"]

requires = [
    "bioimage-cpp",
    "numpy>=2.0",
    "h5py",
    "imageio",
    "pooch",
    "requests",
    "scikit-image",
    "scikit-learn",
    "skan",
    "zarr",
]


# optional dependencies for setuptools
extras = {
    "n5": "pyn5",
    "cloud": "intern"
}

# dependencies only available via conda,
# we still collect them here, because the conda recipe
# gets its requirements from setuptools.
conda_only = ["z5py"]

# collect all dependencies for conda
conda_exclude = [
    "pyn5"  # pyn5 is not available on conda (and not needed due to z5py)
]
conda_all = conda_only + [v for v in extras.values() if v not in conda_exclude]
extras["conda_all"] = conda_all


setup(
    name="python-elf",
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
