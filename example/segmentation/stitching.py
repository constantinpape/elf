import napari

from elf.segmentation.stitching import stitch_segmentation
from skimage.data import binary_blobs
from skimage.measure import label


def connected_components(input_, block_id=None):
    segmentation = label(input_)
    return segmentation.astype("uint32")


def create_test_data(size=1024):
    data = binary_blobs(size, blob_size_fraction=0.1, volume_fraction=0.2)
    return data


def main():
    data = create_test_data(size=1024)

    # compute the segmentation using tiling and stitching
    tile_shape = (256, 256)
    tile_overlap = (32, 32)
    seg_stitched, seg_tiles = stitch_segmentation(
        data, connected_components, tile_shape, tile_overlap, return_before_stitching=True
    )

    # compute the segmentation based on connected components without any stitching
    seg_full = connected_components(data)

    # check the results visually
    v = napari.Viewer()
    v.add_image(data, name="image")
    v.add_labels(seg_full, name="segmentation")
    v.add_labels(seg_stitched, name="stitched segmentation")
    v.add_labels(seg_tiles, name="segmented tiles")
    napari.run()


if __name__ == "__main__":
    main()
