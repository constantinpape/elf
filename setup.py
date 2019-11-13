import runpy
import itertools
from setuptools import setup, find_packages

__version__ = runpy.run_path('elf/__version__.py')['__version__']

requires = [
    "numpy",
    "imageio",
    "scikit-image",
    "skan"
]

extras = {
    "hdf5": ["h5py"],
    "zarr": ["zarr"],
    "vigra": ["vigra"],  # conda-only
    "nifty": ["nifty"],  # conda-only; name clash w/ different PyPI package
    # "n5": ["pyn5"],  # PyPI-only
}

extras["all"] = list(itertools.chain.from_iterable(extras.values()))

setup(
    name='elf',
    packages=find_packages(exclude=['test']),
    version=__version__,
    author='Constantin Pape',
    install_requires=requires,
    extras_require=extras,
    url='https://github.com/constantinpape/elf',
    license='MIT'
    # we will probably have scripts at some point, so I am leaving this for reference
    # entry_points={
    #     "console_scripts": ["view_container = heimdall.scripts.view_container:main"]
    # },
)
