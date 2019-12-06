import os
import subprocess
from shutil import which
import numpy as np
from .files import open_file


def convert_to_point_list(input_file):
    """ Convert mod file to txt with list of points
        using 'model2point'. Do nothing if it's a txt already.
    """
    prefix, ext = os.path.splitext(input_file)
    # convert to tif file if this is .mod file,
    # needs 'model2point'
    if ext == '.mod':
        mod2p = which('model2point')
        if mod2p is None:
            raise RuntimeError("Need model2point in order to convert .mod file to point list")
        tmp_file = prefix + '.txt'
        subprocess.run(['model2point', '-ob', input_file, tmp_file])
        return tmp_file
    # if this is a txt file already, we do nothing
    elif ext == '.txt':
        return input_file
    else:
        raise ValueError("Expect input file format to be .mod or .txt, got %s instead" % ext)


def _dtype_from_max(max_val):
    dtypes = ('uint8', 'uint16', 'uint32', 'uint64')
    for dtype in dtypes:
        if max_val <= np.iinfo(dtype):
            return max_val
    raise ValueError("Max label id exceeds %i and is not supported" % np.iinfo('uint64'))


def _check_dtype(dtype, max_val):
    dmax = np.iinfo(dtype)
    if max_val > dmax:
        raise ValueError("Max label id exceeds dtype")


def _save_to_tensor(point_list, output_path, output_name,
                    shape, chunks, dtype):

    # determine the label ids and the suited datatype
    label_ids = np.unique(point_list[:, 0])
    if dtype is None:
        dtype = _dtype_from_max(label_ids.max())
    else:
        _check_dtype(dtype, label_ids.max())

    ndim = point_list.shape[1:]
    if chunks is None:
        chunks = (64,) * ndim
    chunks_ = tuple(min(ch, sh) for ch, sh in zip(chunks, shape))

    with open_file(output_path, 'a') as f:
        # make the output dataset
        ds = f.create_dataset(output_name, shape, compression='gzip', chunks=chunks_,
                              dtype=dtype)

        # iterate over the label ids and write to the output dataset
        for label_id in label_ids:
            # find all coordinates for this label id
            label_rows = point_list[:, 0] == label_ids
            coords = point_list[:, 1:][label_rows]
            # compute the bounding box
            bounding_box = tuple(slice(int(coords[:, i].min()),
                                       int(coords[:, i + 1].max() + 1)) for i in range(ndim))

            # make the label mask for the bounding box
            bshape = tuple(bb.stop - bb.start for bb in bounding_box)
            label_mask = np.zeros(bshape, dtype=dtype)
            # coords to np.where format
            coords = tuple(coords[:, i] for i in range(ndim))
            label_mask[coords] = label_id

            # write the label mask into the bounding box
            ds[bounding_box] = label_mask


def _to_numpy(point_list, shape, dtype):
    # determine the label ids and the suited datatype
    label_ids = np.unique(point_list[:, 0])
    if dtype is None:
        dtype = _dtype_from_max(label_ids.max())
    else:
        _check_dtype(dtype, label_ids.max())
    ndim = point_list.shape[1:]

    # make the output array
    out = np.zeros(shape, dtype=dtype)

    # TODO this could be completely vectorized
    # iterate over the label ids and write to out
    for label_id in label_ids:
        # find all coordinates for this label id
        label_rows = point_list[:, 0] == label_ids
        coords = point_list[:, 1:][label_rows]
        # coords to np.where format
        coords = tuple(coords[:, i] for i in range(ndim))

        # write the label mask into the bounding box
        out[coords] = label_id
    return out


def _get_shape(shape, point_list):
    max_coordinates = np.max(point_list[1:], axis=1).astype('uint64')
    if shape is None:
        shape = tuple(max_coordinates + 1)
    else:
        if (len(shape) != len(max_coordinates)) or not all(sh > ma for sh, ma in zip(shape, max_coordinates)):
            raise ValueError("Invalid shape argument.")
    return shape


def imod_to_file(input_path, output_path, output_name,
                 shape=None, chunks=None, dtype=None):
    """ Save labels stored in .mod file to a tensor stored on disc.

    Arguments:
        input_file [str] - file path to .mod file (or to .txt file from model2points)
        output_file [str] - output file path, supports the following file formats: hdf5, n5 or zarr.
            The file format will be inferred from the file extension.
        output_name [str] - name of the output dataset.
        shape [tuple] - shape of the output dataset. If None, will use the max coordinates (default: None)
        chunks [tuple] - chunk of the output dataset. If None, will use default chunking (default: None)
        dtype [str or np.dtype] - datatype of the output daaset.
            If None, the smallest possible dtype will be chosen (default: None)
    """
    # make sure the input file is a point list
    in_path_ = convert_to_point_list(input_path)

    # load the point list and get / validate teh shape
    point_list = np.genfromtxt(in_path_)
    shape_ = _get_shape(shape, point_list)
    _save_to_tensor(point_list, output_path, output_name,
                    shape_, chunks, dtype)


def imod_to_numpy(input_path, shape=None, dtype=None):
    """ Save labels stored in .mod file to a tensor stored on disc.

    Arguments:
        input_file [str] - file path to .mod file (or to .txt file from model2points)
        shape [tuple] - shape of the output dataset. If None, will use the max coordinates (default: None)
        dtype [str or np.dtype] - datatype of the output daaset.
            If None, the smallest possible dtype will be chosen (default: None)
    """
    # make sure the input file is a point list
    in_path_ = convert_to_point_list(input_path)

    # load the point list and get / validate teh shape
    point_list = np.genfromtxt(in_path_)
    shape_ = _get_shape(shape, point_list)

    # make numpy array and return it
    return _to_numpy(point_list, shape_, dtype)


# TODO
def imod_to_bdv():
    pass
