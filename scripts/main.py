import timeit
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np

import copybench


def runtest(
    m: int, n: int, dtype: np.dtype, order: str, f: Callable, nRuns: int
) -> float:
    rng = np.random.default_rng(seed=42)
    a = rng.normal(size=(m, n)).astype(dtype, order=order)
    a_T = f(a)

    assert np.all(a.T == a_T)

    timer = timeit.Timer(lambda: f(a))
    return min(timer.repeat(repeat=5, number=nRuns)) / nRuns


def main():
    funcs = [copybench.single_copy, copybench.double_copy]
    orderings = ["C", "F"]

    sizes = [10, 12, 14, 16, 20, 26, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    nRuns = 10

    res = dict()

    for order in orderings:
        for func in funcs:
            times = [
                runtest(size, size, np.float64, order, func, nRuns) for size in sizes
            ]
            res[func.__name__ + f", {order}-ordering"] = times

    fig, ax = plt.subplots()
    fig.set_size_inches(8.5, 5.5)

    for func, times in res.items():
        color = "tab:blue" if "C-ordering" in func else "tab:orange"
        marker = "o" if "single" in func else "x"
        label = (
            ("double copy" if "double" in func else "single copy")
            + ", "
            + func.split(", ")[-1]
        )
        ax.loglog(sizes, times, c=color, marker=marker, lw=2, label=label)

    ax.set(
        xlabel="matrix dimension",
        ylabel="Average execution time [s]",
        title="Comparison of execution times of copying slices",
    )
    ax.legend()
    ax.grid(True)
    ax.autoscale(tight=True, axis="x")
    fig.tight_layout()

    plt.show(block=True)


if __name__ == "__main__":
    plt.ion()
    main()
