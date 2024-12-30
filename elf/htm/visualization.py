import string
from typing import Dict, List, Optional, Tuple, Union

import napari
import numpy as np

from napari.experimental import link_layers


def parse_wells(well_names, zero_based):
    """@private
    """
    well_x, well_y = [], []
    for name in well_names:
        letter, y = name[0], int(name[1:])
        if not zero_based:
            y -= 1
        assert y >= 0
        x = string.ascii_uppercase.index(letter)
        well_x.append(x)
        well_y.append(y)
    assert len(well_x) == len(well_y) == len(well_names)
    well_positions = {name: (x, y) for name, x, y in zip(well_names, well_x, well_y)}
    well_start = (min(well_x), min(well_y))
    well_stop = (max(well_x) + 1, max(well_y) + 1)
    return well_positions, well_start, well_stop


def get_world_position(well_x, well_y, pos, well_shape, well_spacing, site_spacing, shape):
    """@private
    """
    unraveled = np.unravel_index([pos], well_shape)
    pos_x, pos_y = unraveled[0][0], unraveled[1][0]
    i = well_x * well_shape[0] + pos_x
    j = well_y * well_shape[1] + pos_y
    # print(well_x, well_y, pos, ":", i, j)

    sx = shape[-2]
    sy = shape[-1]
    x = sx * i + i * site_spacing + well_x * well_spacing
    y = sy * j + j * site_spacing + well_y * well_spacing

    ndim_non_spatial = len(shape) - 2
    world_pos = ndim_non_spatial * [0] + [x, y]
    return world_pos


def add_grid_sources(
    grid_sources, well_positions, well_shape, well_spacing, site_spacing, add_source, source_settings=None
):
    """@private
    """
    if source_settings is None:
        source_settings = {}
    for channel_name, well_sources in grid_sources.items():
        shape = None
        settings = source_settings.get(channel_name, {})
        channel_layers = []
        for well_name, sources in well_sources.items():
            well_x, well_y = well_positions[well_name]
            for pos, source in enumerate(sources):
                layer = add_source(source, name=f"{channel_name}_{well_name}_{pos}", **settings)
                if shape is None:
                    shape = source.shape
                    if "scale" in settings:
                        scale = settings["scale"]
                        assert len(scale) == len(shape)
                        shape = tuple(sc * sh for sc, sh in zip(shape, scale))
                else:
                    assert source.shape == shape, f"{source.shape}, {shape}"
                world_pos = get_world_position(well_x, well_y, pos, well_shape, well_spacing, site_spacing, shape)
                layer.translate = world_pos
                channel_layers.append(layer)
        link_layers(channel_layers)
    return shape


def _add_layout(viewer, name, box_names, boxes, edge_width, measurements=None, color="coral"):
    properties = {"names": box_names}
    text = "{names}"
    if measurements is not None:
        text += " - "
        for measure_name, measure_values in measurements.items():
            this_measurements = [measure_values[box_name] for box_name in box_names]
            properties[measure_name] = this_measurements
            if isinstance(this_measurements[0], float):
                text += f"{measure_name}: {{{measure_name}:0.2f}},"
            else:
                text += f"{measure_name}: {{{measure_name}}},"
        # get rid of the last comma
        text = text[:-1]
    text_properties = {
        "text": text,
        "anchor": "upper_left",
        "translation": [-5, 0],
        "size": 32,
        "color": "coral"
    }
    viewer.add_shapes(boxes,
                      name=name,
                      properties=properties,
                      text=text_properties,
                      shape_type="rectangle",
                      edge_width=edge_width,
                      edge_color="coral",
                      face_color="transparent")


def add_plate_layout(
    viewer, well_names, well_positions, well_shape, well_spacing, site_spacing, shape,
    measurements=None
):
    """@private
    """
    well_boxes = []
    for well_name in well_names:
        well_x, well_y = well_positions[well_name]
        xmin, ymin = get_world_position(
            well_x, well_y, 0, well_shape, well_spacing, site_spacing, shape
        )[-2:]
        xmin -= well_spacing // 2
        ymin -= well_spacing // 2

        xmax, ymax = get_world_position(
            well_x + 1, well_y + 1, 0, well_shape, well_spacing, site_spacing, shape
        )[-2:]
        xmin -= well_spacing // 2
        ymin -= well_spacing // 2

        well_boxes.append(np.array([[xmin, ymin], [xmax, ymax]]))
    _add_layout(viewer, "wells", well_names, well_boxes, well_spacing // 2,
                measurements=measurements)


def set_camera(viewer, well_start, well_stop, well_shape, well_spacing, site_spacing, shape):
    """@private
    """
    pix_start = get_world_position(
        well_start[0], well_start[1], 0, well_shape, well_spacing, site_spacing, shape
    )[-2:]
    pix_stop = get_world_position(
        well_stop[0], well_stop[1], 0, well_shape, well_spacing, site_spacing, shape
    )[-2:]
    camera_center = tuple((start + stop) // 2 for start, stop in zip(pix_start, pix_stop))
    viewer.camera.center = camera_center

    # zoom out so that we see all wells
    plate_extent = [wstop - wstart for wstart, wstop in zip(well_start, well_stop)]
    max_extent = max(plate_extent)
    max_len = well_shape[np.argmax(plate_extent)]
    viewer.camera.zoom /= (max_len * max_extent)


def view_plate(
    image_data: Dict[str, Dict[str, List[np.ndarray]]],
    label_data: Optional[Dict[str, Dict[str, List[np.ndarray]]]] = None,
    image_settings: Optional[Dict[str, Dict]] = None,
    label_settings: Optional[Dict[str, Dict]] = None,
    well_measurements: Optional[Dict[str, Dict[str, Union[float, int, str]]]] = None,
    well_shape: Optional[Tuple[int, int]] = None,
    zero_based: bool = True,
    well_spacing: int = 16,
    site_spacing: int = 4,
    show: bool = True,
) -> napari.Viewer:
    """Visualize data from a multi-well plate using napari.

    Args:
        image_data: Dict of image sources, each dict must map the channel names to
            a dict which maps the well names (e.g. A1, B3) to
            the image data for this well (one array per well position).
        label_data: Dict of label sources, each dict must map the label name to
            a dict which maps the well names (e.g. A1, B3) to
            the label data for this well (one array per well position).
        image_settings: Image settings for the channels.
        label_settings: Settings for the label layers.
        well_measurements: Measurements associated with the wells.
        well_shape: the 2D shape of a well in terms of images, if not given will be derived.
            Well shape can only be derived for square wells and must be passed otherwise.
        zero_based: Whether the well indexing is zero-based.
        well_sources: Spacing between wells, in pixels.
        site_spacing: Spacing between sites, in pixels.
        show: Whether to show the viewer.

    Returns:
        The napari viewer.
    """
    # find the number of positions per well
    first_channel_sources = next(iter(image_data.values()))
    pos_per_well = len(next(iter(first_channel_sources.values())))

    # find the well shape
    if well_shape is None:  # well shape can only be derived for square wells
        assert pos_per_well in (1, 4, 9, 25, 36, 49), f"well is not square: {pos_per_well}"
        well_len = int(np.sqrt(pos_per_well))
        well_shape = (well_len, well_len)
    else:
        assert len(well_shape) == 2
        pos_per_well_exp = np.prod(list(well_shape))
        assert pos_per_well_exp == pos_per_well, f"{pos_per_well_exp} != {pos_per_well}"

    def process_sources(sources, well_names):
        for well_sources in sources.values():
            # make sure all wells have the same number of label
            n_pos_well = [len(sources) for sources in well_sources.values()]
            assert all(n_pos == pos_per_well for n_pos in n_pos_well), f"{pos_per_well} != {n_pos_well}"
            well_names.extend(list(well_sources.keys()))
        return well_names

    # find the well names for all sources
    well_names = process_sources(image_data, [])
    if label_data is not None:
        well_names = process_sources(label_data, well_names)
    well_names = list(set(well_names))
    well_names.sort()

    # compute the well extent and well positions
    well_positions, well_start, well_stop = parse_wells(well_names, zero_based)
    assert len(well_positions) == len(well_names)

    # start the veiwer and add all sources
    viewer = napari.Viewer()
    shape = add_grid_sources(
        image_data, well_positions, well_shape, well_spacing, site_spacing, viewer.add_image, image_settings
    )
    if label_data is not None:
        add_grid_sources(
            label_data, well_positions, well_shape, well_spacing, site_spacing, viewer.add_labels, label_settings
        )

    # add shape layer corresponding to the well positions
    add_plate_layout(
        viewer, well_names, well_positions, well_shape, well_spacing, site_spacing, shape,
        measurements=well_measurements
    )

    # set the camera so that the initial view is centered around the existing wells
    # and zoom out so that the central well is fully visible
    set_camera(viewer, well_start, well_stop, well_shape, well_spacing, site_spacing, shape)

    if show:
        napari.run()
    return viewer


def add_positional_sources(positional_sources, positions, add_source, source_settings=None):
    """@private
    """
    if source_settings is None:
        source_settings = {}
    for channel_name, sources in positional_sources.items():
        settings = source_settings.get(channel_name, {})
        channel_layers = []
        for sample, source in sources.items():
            layer = add_source(source, name=f"{channel_name}_{sample}", **settings)
            position = positions[sample]
            if len(source.shape) > len(position):
                ndim_non_spatial = len(source.shape) - len(position)
                position = ndim_non_spatial * [0] + list(position)
            layer.translate = list(position)
            channel_layers.append(layer)
            shape = source.shape
        link_layers(channel_layers)
    return shape


def set_camera_positional(viewer, positions, shape):
    """@private
    """
    coords = list(positions.values())
    y = [coord[0] for coord in coords]
    x = [coord[1] for coord in coords]
    ymin, ymax = min(y), max(y)
    xmin, xmax = min(x), max(x)

    camera_center = [(ymax - ymin) // 2, (xmax - xmin) // 2]
    viewer.camera.center = camera_center

    extent = (ymax - ymin, xmax - xmin)
    max_extent = max(extent)
    zoom_out = max_extent / shape[np.argmax(extent)]
    viewer.camera.zoom /= zoom_out


def add_positional_layout(viewer, positions, shape, measurements=None, spacing=16):
    """@private
    """
    boxes = []
    sample_names = []
    for sample, position in positions.items():
        ymin, xmin = position
        ymax, xmax = ymin + shape[0], xmin + shape[1]

        xmin -= spacing
        ymin -= spacing
        xmax += spacing
        ymax += spacing

        boxes.append(np.array([[ymin, xmin], [ymax, xmax]]))
        sample_names.append(sample)
    _add_layout(viewer, "samples", sample_names, boxes, spacing // 2, measurements=measurements)


def view_positional_images(
    image_data: Dict[str, Dict[str, np.ndarray]],
    positions: Dict[str, Tuple[int, int]],
    label_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    image_settings: Optional[Dict[str, Dict]] = None,
    label_settings: Optional[Dict[str, Dict]] = None,
    sample_measurements: Optional[Dict[str, Dict[str, Union[float, int, str]]]] = None,
    show: bool = True,
) -> napari.Viewer:
    """Similar to `view_plate`, but using position data to place the images.

    Args:
        image_data: The image data (outer dict is channels, inner is samples).
        positions: The position for each sample.
        label_data: The label data (outer dict is channels, inner is sample).
        image_settings: Image settings for the channels.
        label_settings: Settings for the label data.
        sample_measurements: Measurements associated with the samples.
        show: Whether to show the viewer.

    Returns:
        The napari viewer.
    """
    all_samples = []
    for sources in image_data.values():
        all_samples.extend(list(sources.keys()))
    if label_data is not None:
        for sources in label_data.values():
            all_samples.extend(list(sources.keys()))

    # make sure we have positional data for all the samples
    assert all(sample in positions for sample in all_samples)

    viewer = napari.Viewer()

    shape = add_positional_sources(image_data, positions, viewer.add_image, image_settings)
    if label_data is not None:
        add_positional_sources(label_data, positions, viewer.add_labels, label_settings)

    add_positional_layout(viewer, positions, shape, measurements=sample_measurements)

    set_camera_positional(viewer, positions, shape)

    if show:
        napari.run()
    return viewer
