from .knossos_wrapper import KnossosFile

# Kudos to @clbarnes for the extension design
FILE_CONSTRUCTORS = {}


def register_filetype(constructor, *extensions):
    FILE_CONSTRUCTORS.update({ext.lower(): constructor for ext in extensions
                              if ext not in FILE_CONSTRUCTORS})


# add hdf5 extensions if we have h5py
try:
    import h5py
    register_filetype(h5py.File, ".h5", ".hdf", ".hdf5")
except ImportError:
    h5py = None

# add n5 and zarr extensions if we have z5py
try:
    import z5py
    register_filetype(z5py.File, ".n5", ".zarr", ".zr")
except ImportError:
    z5py = None

# TODO
# - add zarr extensions if we have zarr-python as a backup to z5py

# Are there any typical knossos extensions?
# add knossos (no extension)
register_filetype(KnossosFile, '')
