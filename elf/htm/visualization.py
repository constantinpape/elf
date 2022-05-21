import string

import napari
import numpy as np

from napari.experimental import link_layers


def parse_wells(well_names, zero_based):
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


def get_world_position(
    well_x, well_y, pos, well_len, well_spacing, site_spacing, shape
):
    i = well_x * well_len + pos // well_len
    j = well_y * well_len + pos % well_len
    # print(well_x, well_y, pos, ":", i, j)

    sx = shape[-2]
    sy = shape[-1]
    x = sx * i + i * site_spacing + well_x * well_spacing
    y = sy * j + j * site_spacing + well_y * well_spacing

    ndim_non_spatial = len(shape) - 2
    world_pos = ndim_non_spatial * [0] + [x, y]
    return world_pos


def add_grid_sources(
    grid_sources, well_positions, well_len, well_spacing, site_spacing, add_source, source_settings=None
):
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
                world_pos = get_world_position(well_x, well_y, pos, well_len, well_spacing, site_spacing, shape)
                # print(well_name, pos, ":", world_pos)
                layer.translate = world_pos
                channel_layers.append(layer)
        link_layers(channel_layers)
    return shape


def add_plate_layout(
    viewer, well_names, well_positions, well_len, well_spacing, site_spacing, shape
):
    well_boxes = []
    for well_name in well_names:
        well_x, well_y = well_positions[well_name]
        xmin, ymin = get_world_position(
            well_x, well_y, 0, well_len, well_spacing, site_spacing, shape
        )[-2:]
        xmin -= well_spacing // 2
        ymin -= well_spacing // 2

        xmax, ymax = get_world_position(
            well_x + 1, well_y + 1, 0, well_len, well_spacing, site_spacing, shape
        )[-2:]
        xmin -= well_spacing // 2
        ymin -= well_spacing // 2

        well_boxes.append(np.array([[xmin, ymin], [xmax, ymax]]))

    properties = {"names": well_names}
    text_properties = {
        "text": "{names}",
        "anchor": "upper_left",
        "translation": [-5, 0],
        "size": 32,
        "color": "coral"
    }

    viewer.add_shapes(well_boxes,
                      name="wells",
                      properties=properties,
                      text=text_properties,
                      shape_type="rectangle",
                      edge_width=well_spacing // 2,
                      edge_color="coral",
                      face_color="transparent")


def set_camera(viewer, well_start, well_stop, well_len, well_spacing, site_spacing, shape):
    pix_start = get_world_position(
        well_start[0], well_start[1], 0, well_len, well_spacing, site_spacing, shape
    )[-2:]
    pix_stop = get_world_position(
        well_stop[0], well_stop[1], 0, well_len, well_spacing, site_spacing, shape
    )[-2:]
    camera_center = tuple((start + stop) // 2 for start, stop in zip(pix_start, pix_stop))
    viewer.camera.center = camera_center

    # zoom out so that we see all wells
    max_plate_width = max([wstop - wstart for wstart, wstop in zip(well_start, well_stop)])
    viewer.camera.zoom /= (well_len * max_plate_width)


# TODO enable non-square wells
def view_plate(
    image_data,
    label_data=None,
    image_settings=None,
    label_settings=None,
    zero_based=True,
    well_spacing=16,
    site_spacing=4,
):
    """Visualize data from a multi-well plate using napari.

    Args:
        image_data dict[str, [dict[str, list[np.array]]]]: list of image sources,
            each list contains a dict which maps the well names (e.g. A1, B3) to
            the image data for this well (one array per well position)
        label_data dict[str, [dict[str, list[np.array]]]]: list of label sources,
            each list contains a dict which maps the well names (e.g. A1, B3) to
            the label data for this well (one array per well position) (default: None)
        image_settings dict[str, dict]: image settings for the channels (default: None)
        label_settings dict[str, dict]: settings for the label channels (default: None)
        zero_based bool: whether the well indexing is zero-based (default: True)
        well_sources int: spacing between wells, in pixels (default: 12)
        site_spacing int: spacing between sites, in pixels (default: 4)
    """
    # find the number of positions per well
    first_channel_sources = next(iter(image_data.values()))
    pos_per_well = len(next(iter(first_channel_sources.values())))
    # only square number of wells allowed
    assert pos_per_well in (4, 9, 25, 36)
    well_len = int(np.sqrt(pos_per_well))

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
        image_data, well_positions, well_len, well_spacing, site_spacing, viewer.add_image, image_settings
    )
    if label_data is not None:
        add_grid_sources(
            label_data, well_positions, well_len, well_spacing, site_spacing, viewer.add_labels, label_settings
        )

    # add shape layer corresponding to the well positions
    add_plate_layout(
        viewer, well_names, well_positions, well_len, well_spacing, site_spacing, shape
    )

    # set the camera so that the initial view is centered around the existing wells
    # and zoom out so that the central well is fully visible
    set_camera(viewer, well_start, well_stop, well_len, well_spacing, site_spacing, shape)

    napari.run()


def add_positional_sources(positional_sources, positions, add_source, source_settings=None):
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


def view_positional_images(image_data, positions, label_data=None, image_settings=None, label_settings=None):
    """Similar to 'view_plate', but using position data parsed to the function to place the images

    Args:
        image_data dict[str, dict[str, np.ndarray]]: the image data (outer dict is channels, inner is sample)
        positions [str, tuple]: the position for each sample
        label_data dict[str, dict[str, np.ndarray]]: the label data (outer dict is channels, inner is sample)
        image_settings dict[str, dict]: image settings for the channels (default: None)
        label_settings dict[str, dict]: settings for the label channels (default: None)
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

    set_camera_positional(viewer, positions, shape)

    napari.run()
