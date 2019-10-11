import numpy as np
import z5py
import matplotlib.pyplot as plt
from elf.sampling.apr import apr


def reconstruct_cremi():
    path = '/home/pape/Work/data/cremi/example/sampleA.n5'
    k = 'volumes/raw/s0'
    f = z5py.File(path)
    ds = f[k]
    ds.n_threads = 4

    bb = np.s_[:25, :256, :256]
    raw = ds[bb]
    print(raw.shape)

    recon = apr(raw, smooth=True)

    fig, ax = plt.subplots(2)
    ax[0].imshow(raw[0])
    ax[1].imshow(recon[0])
    plt.show()


if __name__ == '__main__':
    reconstruct_cremi()
