from elf.io import open_file
from elf.segmentation.watershed import blockwise_two_pass_watershed, distance_transform_watershed


def main():
    """Example data from:
    https://drive.google.com/file/d/1E_Wpw9u8E4foYKk7wvx5RPSWvg_NCN7U/view?usp=sharing
    """
    data_path = "/home/pape/Work/data/cremi/sampleA.n5"
    with open_file(data_path, "r") as f:
        ds = f["volumes/boundaries"]
        ds.n_threads = 8
        boundaries = ds[:64, :512, :512]

    block_shape = (32, 256, 256)
    halo = (8, 64, 64)
    ws_blocks, _ = blockwise_two_pass_watershed(boundaries, block_shape, halo,
                                                verbose=True, threshold=0.25, sigma_seeds=2.0)
    assert ws_blocks.shape == boundaries.shape
    ws, _ = distance_transform_watershed(boundaries, threshold=0.25, sigma_seeds=2.0)

    import napari
    v = napari.Viewer()
    v.add_image(boundaries)
    v.add_labels(ws_blocks)
    v.add_labels(ws)
    napari.run()


main()
