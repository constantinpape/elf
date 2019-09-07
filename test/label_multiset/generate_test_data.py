import os
import numpy as np
import z5py


def generate_test_data():
    """ Generate data for checking label multisets against paintera.

    In order to generate the expected data, use paintera-convesion-helper:
    paintera-conversion-helper -d test.n5,range,label -s 2,2,2 -m -1 -o test_paintera.n5
    paintera-conversion-helper -d test.n5,uniform,label -s 2,2,2 -m -1 -o test_paintera.n5
    """

    folder = '../../data/label_multiset'
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, 'test.n5')

    f = z5py.File(path)
    shape = (32,) * 3

    x = np.arange(np.prod(list(shape)), dtype='uint64').reshape(shape)
    f.create_dataset('range', data=x, chunks=shape, compression='gzip')

    x = 2 * np.ones(shape, dtype='uint64')
    f.create_dataset('uniform', data=x, chunks=shape, compression='gzip')


if __name__ == '__main__':
    generate_test_data()
