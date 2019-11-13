import runpy
import itertools
from setuptools import setup, find_packages

__version__ = runpy.run_path("elf/__version__.py")["__version__"]

requires = [
    "numpy",
    "imageio",
    "scikit-image",
    "skan",
    "h5py",
    "zarr"
]

# extras that are only available on conda
extras_conda = ["vigra", "nifty", "z5py"]
# extras that are only available on pip
extras_pip = ["pyn5"]

extras = {"conda": extras_conda, "pip": extras_pip}


setup(
    name="elf",
    packages=find_packages(exclude=["test"]),
    version=__version__,
    author="Constantin Pape",
    install_requires=requires,
    extras_require=extras,
    url="https://github.com/constantinpape/elf",
    license="MIT"
    # we will probably have scripts at some point, so I am leaving this for reference
    # entry_points={
    #     "console_scripts": ["view_container = heimdall.scripts.view_container:main"]
    # },
)
