

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
    from elf.util import set_numpy_threads
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
