import os
import subprocess
import numpy as np
from .transformix_wrapper import _set_ld_library_path


# TODO implement writing to mhd
def write_to_mhd(image, out_path):
    raise NotImplementedError


def ensure_image(image, output_directory, name):
    if isinstance(image, str):
        if not os.path.exists(image):
            raise ValueError(f"Could not find an image at {image}")
        return image
    elif isinstance(image, np.ndarray):
        out_path = os.path.join(output_directory, name)
        write_to_mhd(image, out_path)
        return out_path
    else:
        raise ValueError(f"Expected image to be either a numpy array or filepath, got {type(image)}")


# TODO
def generate_parameter_file():
    pass


def compute_registration(
    fixed_image,
    moving_image,
    output_directory,
    parameter_file,
    elastix_folder,
    fixed_mask=None,
    moving_mask=None
):
    """Compute registration with elastix.

    Arguments:
        fixed_image [str or np.ndarray] - fixed image, path to mhd file or numpy array
        moving_image [str or np.ndarray] - moving image, path to mhd file or numpy array
        output_directory [str] - directory to store the registered image and transformation
        parameter_file [str] - file with parameters for the elastix registration
        elastix_folder [str] - folder with the elastix binary
        fixed_mask [str or np.ndarray] - optional mask for the fixed image (default: None)
        moving_mask [str or np.ndarray] - optional mask for the moving image (default: None)
    """
    os.makedirs(output_directory, exist_ok=True)
    _set_ld_library_path(elastix_folder)

    fixed_image_path = ensure_image(fixed_image, output_directory, 'fixed.mhd')
    moving_image_path = ensure_image(moving_image, output_directory, 'moving.mhd')

    elastix_bin = os.path.join(
        elastix_folder,
        'bin',
        'elastix'
    )
    cmd = [
        elastix_bin,
        '-f', fixed_image_path,
        '-m', moving_image_path,
        '-out', output_directory,
        '-p', parameter_file
    ]
    if fixed_mask is not None:
        fixed_mask_path = ensure_image(fixed_mask, output_directory, 'fixed_mask.mhd')
        cmd.extend(['-fMask', fixed_mask_path])
    if moving_mask is not None:
        moving_mask_path = ensure_image(moving_mask, output_directory, 'moving_mask.mhd')
        cmd.extend(['-mMask', moving_mask_path])

    subprocess.run(cmd)
