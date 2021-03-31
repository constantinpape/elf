import os
import glob

import numpy as np
try:
    import glasbey as glasey_impl
    GLASBEY_FOLDER = os.path.split(glasey_impl.__file__)[0]
except ImportError:
    glasey_impl = None


def glasbey(n_ids, base_palette_name, overwrite_base_palette=False,
            no_black=True, lightness_range=None, chroma_range=None,
            hue_range=None):
    """ Compute glasbey palette for maximally distant colors.

    Wrapper around https://github.com/taketwo/glasbey, based on
    "Glasbey et al. Colour Displays for Categorical Images."

    Arguments:
        n_ids [int] - number of ids, corresponding to entries in the palette.
        base_palette_name [str] - name of the base palette.
        overwrite_base_palette [bool] -
        no_black [bool] -
        lightness_range [tuple] -
        chroma_range [tuple] -
        hue_range [tuple] -
    """
    if glasey_impl is None:
        raise ImportError("Glasbey module is not available")

    palette_folder = os.path.join(GLASBEY_FOLDER, 'palettes')
    palettes = glob.glob(os.path.join(palette_folder, '*.txt'))
    palettes = {os.path.splitext(os.path.split(name)[1])[0]: name for name in palettes}
    if base_palette_name not in palettes:
        palette_names = list(palettes.keys())
        raise ValueError(f"Invalid palette name: {base_palette_name}, must be one of {palette_names}")
    palette = palettes[base_palette_name]

    gb = glasey_impl.Glasbey(base_palette=palette, no_black=no_black,
                             overwrite_base_palette=overwrite_base_palette,
                             lightness_range=lightness_range,
                             chroma_range=chroma_range,
                             hue_range=hue_range)
    new_palette = gb.generate_palette(size=n_ids)
    new_palette = gb.convert_palette_to_rgb(new_palette)
    return np.array(new_palette, dtype='uint8')


def random_colors(n_ids):
    """ Get random colortable."""
    shape = (n_ids, 3)
    return np.random.randint(0, 255, size=shape, dtype='uint8')
