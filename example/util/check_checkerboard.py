import napari
import bioimage_cpp as bic
import numpy as np
from elf.util import divide_blocks_into_checkerboard


def main():
    shape = (512, 512)
    block_shape = (128, 128)
    blocking = bic.utils.Blocking([0, 0], list(shape), list(block_shape))
    blocks_a, blocks_b = divide_blocks_into_checkerboard(blocking)

    data = np.zeros(shape, dtype="uint8")

    for block_id in range(blocking.number_of_blocks):
        block = blocking.get_block(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        data[bb] = 1 if block_id in blocks_a else 2

    v = napari.Viewer()
    v.add_labels(data)
    napari.run()


if __name__ == "__main__":
    main()
