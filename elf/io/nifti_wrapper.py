from collections.abc import Mapping

try:
    import nibabel
except ImportError:
    nibabel = None


class NiftiFile(Mapping):
    def __init__(self, path, mode="r"):
        self.path = path
        self.mode = mode
        if nibabel is None:
            raise AttributeError("nibabel is required for nifti images, but is not installed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._f.close()

    # dummy attrs to be compatible with h5py/z5py/zarr API
    @property
    def attrs(self):
        return {}


# Go to https://nipy.org/nibabel/nifti_images.html for implementation.
# To be aware of when implementing slicing:
# (Pdb) x = vol[:]
# *** TypeError: Cannot slice image objects; consider using `img.slicer[slice]` to generate a sliced image
# (see documentation for caveats) or slicing image array data with `img.dataobj[slice]` or `img.get_fdata()[slice]`
class NiftiDataset:
    def __init__(self, data_object):
        pass
