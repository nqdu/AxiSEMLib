"""Compare Numba derivative loops for C-order time-last and time-first arrays.

Run from the repository root::

    python benchmarks/benchmark_numba_strain_layout.py

All kernels use a fixed 5 x 5 spatial grid. Timings exclude JIT compilation
and layout conversion, matching ``benchmark_strain_layout.c``.
"""

import statistics
import time

import numpy as np
from numba import jit


@jit(nopython=True)
def derivatives_time_last(u: np.ndarray, G: np.ndarray, GT: np.ndarray,
                          out: np.ndarray) -> None:
    """Differentiate ``u(component, eta, xi, time)`` into a time-last array."""
    _, n, _, nt = u.shape
    for component in range(3):
        for j in range(n):
            for i in range(n):
                for t in range(nt):
                    out[component, j, i, 0, t] = 0.0
                    out[component, j, i, 1, t] = 0.0
                for k in range(n):
                    dx = GT[i, k]
                    de = G[k, j]
                    for t in range(nt):
                        out[component, j, i, 0, t] += dx * u[component, j, k, t]
                        out[component, j, i, 1, t] += de * u[component, k, i, t]


@jit(nopython=True)
def derivatives_time_first(u: np.ndarray, G: np.ndarray, GT: np.ndarray,
                           out: np.ndarray) -> None:
    """Differentiate ``u(time, component, eta, xi)`` with a five-step loop."""
    nt = u.shape[0]
    for t in range(nt):
        for component in range(3):
            for j in range(5):
                for i in range(5):
                    dxi = 0.0
                    deta = 0.0
                    for k in range(5):
                        dxi += GT[i, k] * u[t, component, j, k]
                        deta += G[k, j] * u[t, component, k, i]
                    out[t, component, j, i, 0] = dxi
                    out[t, component, j, i, 1] = deta


@jit(nopython=True)
def derivatives_time_first_unrolled(u: np.ndarray, G: np.ndarray,
                                    GT: np.ndarray, out: np.ndarray) -> None:
    """Differentiate time-first data with the five terms written explicitly."""
    nt = u.shape[0]
    for t in range(nt):
        for component in range(3):
            for j in range(5):
                for i in range(5):
                    dxi = GT[i, 0] * u[t, component, j, 0]
                    dxi += GT[i, 1] * u[t, component, j, 1]
                    dxi += GT[i, 2] * u[t, component, j, 2]
                    dxi += GT[i, 3] * u[t, component, j, 3]
                    dxi += GT[i, 4] * u[t, component, j, 4]

                    deta = G[0, j] * u[t, component, 0, i]
                    deta += G[1, j] * u[t, component, 1, i]
                    deta += G[2, j] * u[t, component, 2, i]
                    deta += G[3, j] * u[t, component, 3, i]
                    deta += G[4, j] * u[t, component, 4, i]

                    out[t, component, j, i, 0] = dxi
                    out[t, component, j, i, 1] = deta


def median_time(function, calls: int) -> float:
    """Return median microseconds per call after the kernel has compiled."""
    samples = []
    for _ in range(9):
        start = time.perf_counter()
        for _ in range(calls):
            function()
        samples.append((time.perf_counter() - start) * 1e6 / calls)
    return statistics.median(samples)


def main() -> None:
    rng = np.random.default_rng(77)
    G = rng.standard_normal((5, 5))
    GT = rng.standard_normal((5, 5))
    print('samples  time-last [us]  time-first [us]  unrolled [us]')
    for nt in (64, 512, 2048):
        u_last = rng.standard_normal((3, 5, 5, nt))
        u_first = np.ascontiguousarray(u_last.transpose(3, 0, 1, 2))
        out_last = np.empty((3, 5, 5, 2, nt))
        out_first = np.empty((nt, 3, 5, 5, 2))
        calls = 500 if nt == 64 else 80

        kernels = (
            lambda: derivatives_time_last(u_last, G, GT, out_last),
            lambda: derivatives_time_first(u_first, G, GT, out_first),
            lambda: derivatives_time_first_unrolled(u_first, G, GT, out_first),
        )
        for index, kernel in enumerate(kernels):
            kernel()
            if index:
                np.testing.assert_allclose(
                    out_first.transpose(1, 2, 3, 4, 0), out_last,
                    rtol=1e-12, atol=1e-12,
                )
        timings = [median_time(kernel, calls) for kernel in kernels]
        print(f'{nt:7d}  {timings[0]:14.2f}  {timings[1]:15.2f}'
              f'  {timings[2]:13.2f}')


if __name__ == '__main__':
    main()
