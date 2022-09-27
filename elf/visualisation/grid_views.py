#
# simple grid image views implemented on top of napari
# simplified version of elf.htm
#

import napari
import numpy as np

from napari.experimental import link_layers


def get_position(grid_shape, image_shape, i, spacing):
    unraveled = np.unravel_index([i], grid_shape)
    grid_x, grid_y = unraveled[0][0], unraveled[1][0]
    x = (image_shape[0] + spacing) * grid_x
    y = (image_shape[1] + spacing) * grid_y
    return x, y


def add_grid_sources(name, images, grid_shape, settings, add_source, spacing):
    layers = []
    for i, image in enumerate(images):
        layer = add_source(image, name=f"{name}-{i}", **settings)
        position = get_position(grid_shape, image.shape, i, spacing)
        layer.translate = position
        layers.append(layer)
    link_layers(layers)


def set_camera(viewer, grid_shape, image_shape, spacing):
    # find the full extent in pixels
    extent = [gsh * (ish + spacing) for gsh, ish in zip(grid_shape, image_shape)]

    # set the camera center to the middle
    camera_center = tuple(ext // 2 for ext in extent)
    viewer.camera.center = camera_center

    # zoom out so that we see all images in the grid
    max_len = grid_shape[np.argmax(extent)]
    viewer.camera.zoom /= max_len


def simple_grid_view(image_data, label_data=None, settings=None, grid_shape=None, spacing=16, show=True):
    """
    Args:
        image_data [dict[list[np.ndarray]]] -
        label_data [dict[list[np.ndarray]]] -
        settings [dict] -
        grid_shape [tuple] -
        spacing [int] -
        show [bool] -
    """

    n_images = len(next(iter(image_data.values())))
    image_shape = next(iter(image_data.values()))[0].shape

    # find the grid shape
    if grid_shape is None:  # grid shape can only be derived for square grids
        assert n_images in (1, 4, 9, 16, 25, 36, 49, 64, 81, 100),\
            f"grid is not square or too large: {n_images}"
        grid_len = int(np.sqrt(n_images))
        grid_shape = (grid_len, grid_len)
    assert len(grid_shape) == 2
    assert grid_shape[0] * grid_shape[1] == n_images

    viewer = napari.Viewer()
    for name, images in image_data.items():
        assert len(images) == n_images
        assert all([im.shape == image_shape for im in images])
        this_settings = {} if settings is None else settings.get(name, {})
        add_grid_sources(name, images, grid_shape, this_settings,
                         add_source=viewer.add_image, spacing=spacing)

    if label_data is not None:
        for name, labels in label_data.items():
            assert len(labels) == n_images
            assert all([label.shape == image_shape for label in labels])
            this_settings = {} if settings is None else settings.get(name, {})
            add_grid_sources(name, labels, grid_shape, this_settings,
                             add_source=viewer.add_labels, spacing=spacing)

    set_camera(viewer, grid_shape, image_shape, spacing)

    if show:
        napari.run()
    return viewer
