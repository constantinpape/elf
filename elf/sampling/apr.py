import numpy as np
import pyApr


# TODO parameters
def apr(data, smooth=False):
    dtype = data.dtype
    if dtype == np.dtype('uint8'):
        sampler = pyApr.AprByte()
    elif dtype == np.dtype('uint16'):
        sampler = pyApr.AprShort()
    elif dtype == np.dtype('float32'):
        sampler = pyApr.AprFloat()
    else:
        raise ValueError("Datatype %s not supported, must be one of uint8, uint16 or float32"
                         % str(dtype))
    sampler.get_apr_from_array(data)
    if smooth:
        reconstruction = sampler.reconstruct()
    else:
        reconstruction = sampler.reconstruct_smooth()
    reconstruction = np.array(reconstruction, copy=False)
    return reconstruction
