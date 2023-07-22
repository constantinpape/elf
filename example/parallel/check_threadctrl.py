import os
import ctypes


def set_numpy_threads(n_threads):
    """ Set the number of threads numpy exposes to its underlying linalg library.

    This is now deprecated in favor of threadctl and was moved here from elf.util.

    This needs to be called BEFORE the numpy import and sets the number
    of threads statically.
    Based on answers in https://github.com/numpy/numpy/issues/11826.
    """

    # set number of threads for mkl if it is used
    try:
        import mkl
        mkl.set_num_threaads(n_threads)
    except Exception:
        pass

    for name in ["libmkl_rt.so", "libmkl_rt.dylib", "mkl_Rt.dll"]:
        try:
            mkl_rt = ctypes.CDLL(name)
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(n_threads)))
        except Exception:
            pass

    # set number of threads in all possibly relevant environment variables
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)


def check_normal():
    import time
    import numpy as np

    x = np.random.rand(2048, 2048)
    N = 10

    ts = []
    print("Start")
    for _ in range(N):
        t0 = time.time()
        x @ x
        ts.append(time.time() - t0)
    print(min(ts))


def check_numpy_threads():
    set_numpy_threads(1)
    import time
    import numpy as np

    x = np.random.rand(2048, 2048)
    N = 10

    ts = []
    print("Start")
    for _ in range(N):
        t0 = time.time()
        x @ x
        ts.append(time.time() - t0)
    print(min(ts))


def check_threadctrl():
    import time
    from threadpoolctl import threadpool_limits
    import numpy as np

    x = np.random.rand(2048, 2048)
    N = 10

    ts = []
    print("Start")
    with threadpool_limits(limits=1):
        for _ in range(N):
            t0 = time.time()
            x @ x
            ts.append(time.time() - t0)
    print(min(ts))


def main():
    # time: 0.11289644241333008
    # check_normal()

    # time: 0.4072251319885254
    # check_numpy_threads()

    # It works!
    # time: 0.4088623523712158
    check_threadctrl()


if __name__ == "__main__":
    main()
