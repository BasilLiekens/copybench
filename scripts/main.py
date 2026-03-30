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
    colors = {
        copybench.single_copy: "tab:blue",
        copybench.double_copy: "tab:orange",
        copybench.blocked_copy: "tab:green",
    }
    labels = {
        copybench.single_copy: "single copy (naive)",
        copybench.double_copy: "double copy",
        copybench.blocked_copy: "single copy (blocked)",
    }
    markers = {"C": "o", "F": "x"}

    funcs = [copybench.single_copy, copybench.double_copy, copybench.blocked_copy]
    orderings = ["C", "F"]

    sizes = [10, 12, 14, 16, 20, 26, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    nRuns = 10

    fig, ax = plt.subplots()
    fig.set_size_inches(8.5, 5.5)

    for order in orderings:
        for func in funcs:
            times = [
                runtest(size, size, np.float64, order, func, nRuns) for size in sizes
            ]
            ax.loglog(
                sizes,
                times,
                c=colors[func],
                marker=markers[order],
                lw=2,
                label=f"{labels[func]}, {order}-ordered",
            )

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
