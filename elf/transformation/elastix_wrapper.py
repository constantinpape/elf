import os
import subprocess
from typing import Optional, Union

import numpy as np
from .transformix_wrapper import _set_ld_library_path


# TODO implement writing to mhd
def write_to_mhd(image, out_path):
    """@private
    """
    raise NotImplementedError


def ensure_image(image, output_directory, name):
    """@private
    """
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


def generate_parameter_file():
    """@private
    """
    pass


def compute_registration(
    fixed_image: Union[str, np.ndarray],
    moving_image: Union[str, np.ndarray],
    output_directory: str,
    parameter_file: str,
    elastix_folder: str,
    fixed_mask: Optional[Union[str, np.ndarray]] = None,
    moving_mask: Optional[Union[str, np.ndarray]] = None,
):
    """Compute registration with elastix.

    Args:
        fixed_image: Fixed image, path to mhd file or numpy array.
        moving_image: Moving image, path to mhd file or numpy array.
        output_directory: Directory to store the registered image and transformation.
        parameter_file: File with parameters for the elastix registration.
        elastix_folder: Folder with the elastix binary.
        fixed_mask: Optional mask for the fixed image.
        moving_mask: Optional mask for the moving image.
    """
    os.makedirs(output_directory, exist_ok=True)
    _set_ld_library_path(elastix_folder)

    fixed_image_path = ensure_image(fixed_image, output_directory, "fixed.mhd")
    moving_image_path = ensure_image(moving_image, output_directory, "moving.mhd")

    elastix_bin = os.path.join(elastix_folder, "bin", "elastix")
    cmd = [
        elastix_bin,
        "-f", fixed_image_path,
        "-m", moving_image_path,
        "-out", output_directory,
        "-p", parameter_file
    ]
    if fixed_mask is not None:
        fixed_mask_path = ensure_image(fixed_mask, output_directory, "fixed_mask.mhd")
        cmd.extend(["-fMask", fixed_mask_path])
    if moving_mask is not None:
        moving_mask_path = ensure_image(moving_mask, output_directory, "moving_mask.mhd")
        cmd.extend(["-mMask", moving_mask_path])

    subprocess.run(cmd)
