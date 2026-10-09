"""Compare the C-order Numba kernels with the original compiled Fortran code.

Run from the repository root after building ``lib/libsem``::

    python benchmarks/benchmark_jit_sem.py

Times exclude JIT compilation and include the normal Python wrappers.
"""

from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axisemlib.sem_funcs import (  # noqa: E402
    _mapping_jacobian,
    find_theta,
    inside_element,
    lagrange_interpol_2D_td,
    strain_td,
)

try:
    from lib import libsem  # noqa: E402
except ImportError as exc:
    raise SystemExit('Build lib/libsem before running this comparison') from exc


def fortran_inside(s, z, nodes, element_type, tolerance):
    nodes = np.require(nodes, dtype=np.float64, requirements=['F_CONTIGUOUS'])
    return libsem.inside_element_warp(s, z, nodes, element_type, tolerance)


def fortran_interpolate(points1, points2, coefficients, x1, x2):
    points1 = np.require(points1, dtype=np.float64, requirements=['F_CONTIGUOUS'])
    points2 = np.require(points2, dtype=np.float64, requirements=['F_CONTIGUOUS'])
    coefficients = np.require(coefficients, dtype=np.float64,
                              requirements=['F_CONTIGUOUS'])
    return libsem.lagrange_2D(points1, points2, coefficients, x1, x2)


def fortran_theta(xi, eta, nodes, element_type):
    xi = np.require(xi, dtype=np.float64, requirements=['F_CONTIGUOUS'])
    eta = np.require(eta, dtype=np.float64, requirements=['F_CONTIGUOUS'])
    nodes = np.require(nodes, dtype=np.float64, requirements=['F_CONTIGUOUS'])
    return libsem.find_theta(xi, eta, nodes, element_type)


def fortran_strain(u, G, GT, xi, eta, npol, nsamp, nodes,
                   element_type, axial, stype):
    arrays = [np.require(a, dtype=np.float64, requirements=['F_CONTIGUOUS'])
              for a in (u, G, GT, xi, eta, nodes)]
    result = libsem.strain_td_warp(*arrays, element_type, axial, stype)
    return np.reshape(result, (nsamp, npol + 1, npol + 1, 6), order='F')


def check_equal(name, old, new, *, atol=1e-12):
    if isinstance(old, tuple):
        assert old[0] == new[0], name
        old, new = old[1:], new[1:]
    np.testing.assert_allclose(old, new, rtol=1e-10, atol=atol,
                               err_msg=name)


def median_time(function, calls):
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        for _ in range(calls):
            function()
        samples.append((time.perf_counter() - start) / calls)
    return statistics.median(samples)


def main():
    rng = np.random.default_rng(77)
    r0, r1 = 5.9e6, 6.0e6
    theta0, theta1 = 0.50, 0.55
    nodes = np.array([
        [r0 * np.sin(theta0), r0 * np.cos(theta0)],
        [r0 * np.sin(theta1), r0 * np.cos(theta1)],
        [r1 * np.sin(theta1), r1 * np.cos(theta1)],
        [r1 * np.sin(theta0), r1 * np.cos(theta0)],
    ])
    axial_nodes = np.array([
        [0.0, r0], [3.0e4, r0], [3.0e4, r1], [0.0, r1],
    ])
    gll = np.array([-1.0, -0.6546536707079771, 0.0,
                    0.6546536707079771, 1.0])
    eta = np.array([-1.0, -0.42, 0.13, 0.74, 1.0])
    G = rng.standard_normal((5, 5))
    GT = rng.standard_normal((5, 5))
    u = rng.standard_normal((3, 5, 5, 2048))
    coefficients = rng.standard_normal((5, 5, 2048))
    u_fortran = u.transpose(3, 2, 1, 0)
    coefficients_fortran = coefficients.transpose(2, 1, 0)
    assert u.flags.c_contiguous and u.strides[-1] == u.itemsize
    assert coefficients.flags.c_contiguous and coefficients.strides[-1] == coefficients.itemsize

    # Check all mappings and strain modes against the original Fortran calls.
    for element_type in range(4):
        for xi_point, eta_point in ((0.1, -0.3), (0.7, 0.9), (-0.5, 0.2)):
            s, z = _mapping_jacobian(xi_point, eta_point, nodes, element_type)[:2]
            check_equal('inside_element',
                        fortran_inside(s, z, nodes, element_type, 1e-5),
                        inside_element(s, z, nodes, element_type, 1e-5),
                        atol=1e-10)

        theta = find_theta(gll, eta, nodes, element_type)
        assert theta.flags.c_contiguous
        check_equal('find_theta',
                    fortran_theta(gll, eta, nodes, element_type).T, theta)

        for stype in ('monopole', 'dipole', 'quadpole'):
            strain = strain_td(u, G, GT, gll, eta, 4, 2048,
                               nodes, element_type, False, stype)
            assert strain.flags.c_contiguous and strain.strides[-1] == strain.itemsize
            check_equal('strain_td',
                        fortran_strain(u_fortran, G, GT, gll, eta, 4, 2048,
                                       nodes, element_type, False, stype
                                       ).transpose(3, 2, 1, 0),
                        strain)
            if element_type:
                check_equal('strain_td axial',
                            fortran_strain(u_fortran, G, GT, gll, eta, 4, 2048,
                                           axial_nodes, element_type, True, stype
                                           ).transpose(3, 2, 1, 0),
                            strain_td(u, G, GT, gll, eta, 4, 2048,
                                      axial_nodes, element_type, True, stype))

    # Distinct xi and eta points expose accidental spatial transposes.
    check_equal('interpolation',
                fortran_interpolate(gll, eta, coefficients_fortran, 0.21, -0.13),
                lagrange_interpol_2D_td(gll, eta, coefficients, 0.21, -0.13))
    coefficients_f32 = coefficients.astype(np.float32)
    check_equal('interpolation float32',
                fortran_interpolate(gll, eta,
                                    coefficients_f32.transpose(2, 1, 0), 0.21, -0.13),
                lagrange_interpol_2D_td(gll, eta, coefficients_f32, 0.21, -0.13))

    s, z = _mapping_jacobian(0.2, -0.3, nodes, 0)[:2]
    cases = [
        ('inside_element',
         lambda: fortran_inside(s, z, nodes, 0, 1e-5),
         lambda: inside_element(s, z, nodes, 0, 1e-5), 10000),
        ('find_theta',
         lambda: fortran_theta(gll, eta, nodes, 0),
         lambda: find_theta(gll, eta, nodes, 0), 2000),
        ('interpolate (2048 samples)',
         lambda: fortran_interpolate(gll, eta, coefficients_fortran, 0.21, -0.13),
         lambda: lagrange_interpol_2D_td(gll, eta, coefficients, 0.21, -0.13),
         1000),
    ]
    for stype in ('monopole', 'dipole', 'quadpole'):
        cases.append((f'strain {stype} (2048 samples)',
                      lambda stype=stype: fortran_strain(
                          u_fortran, G, GT, gll, eta, 4, 2048,
                          nodes, 0, False, stype),
                      lambda stype=stype: strain_td(
                          u, G, GT, gll, eta, 4, 2048,
                          nodes, 0, False, stype), 40))

    print('All numerical comparisons passed.')
    print('Operation                              Fortran [us]    JIT [us]  Speedup')
    for name, old, new, calls in cases:
        old()
        new()
        old_time = median_time(old, calls)
        new_time = median_time(new, calls)
        print(f'{name:38s} {old_time * 1e6:11.2f} '
              f'{new_time * 1e6:11.2f} {old_time / new_time:8.2f}x')


if __name__ == '__main__':
    main()
