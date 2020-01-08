

def get_block_shape(data, block_shape):
    if block_shape is None:
        try:
            block_shape = data.chunks
        except AttributeError:
            raise ValueError("If block_shape is not given, the data needs to have a chunk attribute.")
    return block_shape
