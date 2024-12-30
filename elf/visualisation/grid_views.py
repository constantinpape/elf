#
# simple grid image views implemented on top of napari
# simplified version of elf.htm
#
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import napari
    from napari.experimental import link_layers
except ImportError:
    napari = None
    link_layers = None


def get_position(grid_shape, image_shape, i, spacing):
    """@private
    """
    unraveled = np.unravel_index([i], grid_shape)
    grid_x, grid_y = unraveled[0][0], unraveled[1][0]
    x = (image_shape[-2] + spacing) * grid_x
    y = (image_shape[-1] + spacing) * grid_y
    return x, y


def add_grid_sources(name, images, grid_shape, settings, add_source, spacing, is_rgb):
    """@private
    """
    layers = []
    for i, image in enumerate(images):
        layer = add_source(image, name=f"{name}-{i}", **settings)
        position = get_position(grid_shape, image.shape[:-1] if is_rgb else image.shape, i, spacing)
        layer.translate = position
        layers.append(layer)
    link_layers(layers)


def set_camera(viewer, grid_shape, image_shape, spacing):
    """@private
    """
    # find the full extent in pixels
    extent = [gsh * (ish + spacing) for gsh, ish in zip(grid_shape, image_shape[-2:])]

    # set the camera center to the middle
    camera_center = tuple(ext // 2 for ext in extent)
    viewer.camera.center = camera_center

    # zoom out so that we see all images in the grid
    max_len = grid_shape[np.argmax(extent)]
    viewer.camera.zoom /= max_len


# TODO make is_rgb support a dict
def simple_grid_view(
    image_data: Dict[str, List[np.ndarray]],
    label_data: Optional[Dict[str, List[np.ndarray]]] = None,
    settings: Optional[Dict[str, Dict]] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    spacing: int = 16,
    show: bool = True,
    is_rgb: bool = False,
) -> napari.Viewer:
    """Show images in napari using a simple grid view.

    Args:
        image_data: Dictionary with the image data for the grid positions.
        label_data: Dictionary with the label data for the grid positions.
        settings: The napari layer settings for images / labels.
        grid_shape: The shape of the grid. If None, it will be derived from the number of images.
        spacing: The spacing between images in the grid.
        show: Whether to start the napari viewer.
        is_rgb: Whether the image data is rgb.

    Returns:
        The napari viewer.
    """
    assert napari is not None and link_layers is not None, "Requires napari"

    n_images = len(next(iter(image_data.values())))
    image_shape = next(iter(image_data.values()))[0].shape
    if is_rgb:
        assert image_shape[-1] == 3
        image_shape = image_shape[:-1]

    # find the grid shape
    if grid_shape is None:  # grid shape can only be derived for square grids
        assert n_images in (1, 4, 9, 16, 25, 36, 49, 64, 81, 100), \
            f"grid is not square or too large: {n_images}"
        grid_len = int(np.sqrt(n_images))
        grid_shape = (grid_len, grid_len)
    assert len(grid_shape) == 2
    assert grid_shape[0] * grid_shape[1] == n_images

    viewer = napari.Viewer()
    for name, images in image_data.items():
        assert len(images) == n_images
        if is_rgb:
            assert all([im.shape[:-1] == image_shape for im in images])
        else:
            assert all([im.shape == image_shape for im in images])
        this_settings = {} if settings is None else settings.get(name, {})
        add_grid_sources(name, images, grid_shape, this_settings,
                         add_source=viewer.add_image, spacing=spacing,
                         is_rgb=is_rgb)

    if label_data is not None:
        for name, labels in label_data.items():
            assert len(labels) == n_images
            assert all([label.shape == image_shape for label in labels])
            this_settings = {} if settings is None else settings.get(name, {})
            add_grid_sources(name, labels, grid_shape, this_settings,
                             add_source=viewer.add_labels, spacing=spacing,
                             is_rgb=False)

    set_camera(viewer, grid_shape, image_shape, spacing)

    if show:
        napari.run()
    return viewer
