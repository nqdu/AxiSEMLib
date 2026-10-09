"""Numba spectral-element mapping, interpolation, and strain kernels.

The formulas follow the original AxiSEM Fortran kernels by Martin van Driel
and Lion Krischer (LGPL v3). Time-dependent arrays are C contiguous
with time last. Spatial axes are (eta, xi); displacement components are
(s, phi, z), and strain components are (ss, pp, zz, pz, sz, sp).
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def _edge(xi: float, nodes: np.ndarray, left: int, right: int,
          curved: bool) -> tuple[float, float, float, float]:
    """Map one element edge in the meridional plane.

    Parameters
    ----------
    xi : float
        Local edge coordinate in [-1, 1].
    nodes : ndarray, shape (4, 2)
        Element corners, with (s, z) columns in metres.
    left, right : int
        Indices of the two edge corners.
    curved : bool
        Whether the edge follows a spherical arc.

    Returns
    -------
    tuple of float
        (s, z, ds/dxi, dz/dxi) at ``xi``, in metres.
    """
    if not curved:
        s = ((1.0 - xi) * nodes[left, 0] +
             (1.0 + xi) * nodes[right, 0]) / 2.0
        z = ((1.0 - xi) * nodes[left, 1] +
             (1.0 + xi) * nodes[right, 1]) / 2.0
        return (s, z,
                (nodes[right, 0] - nodes[left, 0]) / 2.0,
                (nodes[right, 1] - nodes[left, 1]) / 2.0)

    sl, zl = nodes[left, 0], nodes[left, 1]
    sr, zr = nodes[right, 0], nodes[right, 1]
    rl = np.sqrt(sl * sl + zl * zl)
    rr = np.sqrt(sr * sr + zr * zr)
    tl = np.arccos(zl / rl) if rl != 0.0 else 0.0
    tr = np.arccos(zr / rr) if rr != 0.0 else 0.0
    theta = ((1.0 - xi) * tl + (1.0 + xi) * tr) / 2.0
    dtheta = (tr - tl) / 2.0
    radius = rl
    return (radius * np.sin(theta), radius * np.cos(theta),
            radius * np.cos(theta) * dtheta,
            -radius * np.sin(theta) * dtheta)


@jit(nopython=True, cache=True)
def _mapping_jacobian(xi: float, eta: float, nodes: np.ndarray,
                      element_type: int) -> tuple[float, float, float, float,
                                                   float, float]:
    """Map one element point and compute its Jacobian.

    Parameters
    ----------
    xi, eta : float
        Local coordinates in [-1, 1].
    nodes : ndarray, shape (4, 2)
        Element corners, with (s, z) columns in metres.
    element_type : int
        Element mapping code 0 through 3.

    Returns
    -------
    tuple of float
        (s, z, ds/dxi, ds/deta, dz/dxi, dz/deta), in metres.
    """
    bottom = _edge(xi, nodes, 0, 1, element_type == 0 or element_type == 3)
    top = _edge(xi, nodes, 3, 2, element_type == 0 or element_type == 2)
    low = (1.0 - eta) / 2.0
    high = (1.0 + eta) / 2.0
    s = low * bottom[0] + high * top[0]
    z = low * bottom[1] + high * top[1]
    ds_dxi = low * bottom[2] + high * top[2]
    ds_deta = (top[0] - bottom[0]) / 2.0
    dz_dxi = low * bottom[3] + high * top[3]
    dz_deta = (top[1] - bottom[1]) / 2.0
    return s, z, ds_dxi, ds_deta, dz_dxi, dz_deta


@jit(nopython=True, cache=True)
def _inside_element(s: float, z: float, nodes: np.ndarray,
                    element_type: int, tolerance: float
                    ) -> tuple[bool, float, float]:
    """Locate a cylindrical point inside one element.

    Parameters
    ----------
    s, z : float
        Cylindrical horizontal and axial coordinates in metres.
    nodes : ndarray, shape (4, 2)
        Element corners, with (s, z) columns in metres.
    element_type : int
        Element mapping code 0 through 3.
    tolerance : float
        Allowed excess beyond each local [-1, 1] bound.

    Returns
    -------
    tuple of (bool, float, float)
        Inside flag and dimensionless local coordinates (xi, eta).
    """
    xi = 0.0
    eta = 0.0
    radius_squared = s * s + z * z
    for _ in range(10):
        mapped_s, mapped_z, j11, j12, j21, j22 = _mapping_jacobian(
            xi, eta, nodes, element_type)
        ds = s - mapped_s
        dz = z - mapped_z
        if radius_squared > 0.0 and (ds * ds + dz * dz) / radius_squared < 1e-14:
            break
        determinant = j11 * j22 - j21 * j12
        xi += (j22 * ds - j12 * dz) / determinant
        eta += (-j21 * ds + j11 * dz) / determinant
    inside = (-1.0 - tolerance <= xi <= 1.0 + tolerance and
              -1.0 - tolerance <= eta <= 1.0 + tolerance)
    return inside, xi, eta


@jit(nopython=True, cache=True)
def _find_theta(xi: np.ndarray, eta: np.ndarray, nodes: np.ndarray,
                element_type: int, out: np.ndarray) -> None:
    """Fill a C-order grid of mapped colatitudes.

    Parameters
    ----------
    xi, eta : ndarray
        One-dimensional local node arrays of lengths nxi and neta.
    nodes : ndarray, shape (4, 2)
        Element corners, with (s, z) columns in metres.
    element_type : int
        Element mapping code 0 through 3.
    out : ndarray, shape (neta, nxi)
        Output colatitudes in radians; modified in place.

    Returns
    -------
    None
    """
    for j in range(len(eta)):
        for i in range(len(xi)):
            s, z, _, _, _, _ = _mapping_jacobian(
                xi[i], eta[j], nodes, element_type)
            out[j, i] = np.arccos(z / np.sqrt(s * s + z * z))


@jit(nopython=True, cache=True)
def _interpolation_weights(points: np.ndarray, x: float) -> np.ndarray:
    """Compute one-dimensional Lagrange interpolation weights.

    Parameters
    ----------
    points : ndarray, shape (n,)
        Collocation coordinates.
    x : float
        Target coordinate in the same system as ``points``.

    Returns
    -------
    ndarray, shape (n,)
        Interpolation weights corresponding to ``points``.
    """
    n = len(points)
    weights = np.empty(n, dtype=np.float64)
    for i in range(n):
        weight = 1.0
        for m in range(n):
            if m != i:
                weight *= (x - points[m]) / (points[i] - points[m])
        weights[i] = weight
    return weights


@jit(nopython=True, cache=True)
def _interpolate_time_last(points1: np.ndarray, points2: np.ndarray,
                           coefficients: np.ndarray, x1: float,
                           x2: float) -> np.ndarray:
    """Interpolate a time series on one tensor-product element.

    Parameters
    ----------
    points1 : ndarray, shape (nxi,)
        Local xi nodes.
    points2 : ndarray, shape (neta,)
        Local eta nodes.
    coefficients : ndarray, shape (neta, nxi, nt)
        C-order field values with time fastest.
    x1, x2 : float
        Target xi and eta coordinates.

    Returns
    -------
    ndarray, shape (nt,)
        Interpolated time series.
    """
    weights1 = _interpolation_weights(points1, x1)
    weights2 = _interpolation_weights(points2, x2)
    result = np.zeros(coefficients.shape[2], dtype=np.float64)
    for i in range(len(points1)):
        for j in range(len(points2)):
            weight = weights1[i] * weights2[j]
            for t in range(len(result)):
                result[t] += coefficients[j, i, t] * weight
    return result


@jit(nopython=True, cache=True)
def _strain_time_last(u: np.ndarray, G: np.ndarray, GT: np.ndarray,
                      xi: np.ndarray, eta: np.ndarray, nodes: np.ndarray,
                      element_type: int, axial: bool, mode: int,
                      out: np.ndarray) -> None:
    """Assemble strain from time-dependent modal displacement.

    Parameters
    ----------
    u : ndarray, shape (3, neta, nxi, nt)
        C-order displacement with components (s, phi, z) and time fastest.
    G : ndarray, shape (n, n)
        Eta derivative matrix.
    GT : ndarray, shape (n, n)
        Xi derivative matrix; use GLJ at the axis.
    xi : ndarray, shape (nxi,)
        Local xi nodes.
    eta : ndarray, shape (neta,)
        Local eta nodes.
    nodes : ndarray, shape (4, 2)
        Corner coordinates in metres, with columns (s, z).
    element_type : int
        Element mapping code from 0 through 3.
    axial : bool
        Whether the element touches s = 0.
    mode : int
        Azimuthal mode: 0 monopole, 1 dipole, or 2 quadpole.
    out : ndarray, shape (6, neta, nxi, nt)
        C-order output strain in (ss, pp, zz, pz, sz, sp) order.

    Returns
    -------
    None
        ``out`` is modified in place.
    """
    n, nt = out.shape[1], out.shape[3]
    # Per-point radius and inverse Jacobian: (neta, nxi, 5).
    geometry = np.empty((n, n, 5), dtype=np.float64)
    # One component's xi and eta derivatives, with time contiguous.
    grad = np.empty((2, nt), dtype=np.float64)

    # Compute the geometry once at each point before the component passes.
    for j in range(n):
        for i in range(n):
            s, _, j11, j12, j21, j22 = _mapping_jacobian(
                xi[i], eta[j], nodes, element_type)
            determinant = j11 * j22 - j21 * j12
            geometry[j, i, 0] = s
            geometry[j, i, 1] = j22 / determinant
            geometry[j, i, 2] = -j12 / determinant
            geometry[j, i, 3] = -j21 / determinant
            geometry[j, i, 4] = j11 / determinant

    # Differentiate one component over the element, then add its strain terms.
    for component in range(3):
        if mode == 0 and component == 1:
            continue
        for j in range(n):
            for i in range(n):
                s = geometry[j, i, 0]
                inv11 = geometry[j, i, 1]
                inv12 = geometry[j, i, 2]
                inv21 = geometry[j, i, 3]
                inv22 = geometry[j, i, 4]
                for t in range(nt):
                    grad[0, t] = 0.0
                    grad[1, t] = 0.0
                for k in range(n):
                    dx = GT[i, k]
                    de = G[k, j]
                    for t in range(nt):
                        grad[0, t] += dx * u[component, j, k, t]
                        grad[1, t] += de * u[component, k, i, t]

                # Assemble this component's contribution to Voigt strain.
                for t in range(nt):
                    ds = inv11 * grad[0, t] + inv21 * grad[1, t]
                    dz = inv12 * grad[0, t] + inv22 * grad[1, t]
                    u_over_s = (ds if axial and i == 0
                                else u[component, j, i, t] / s)
                    if component == 0:
                        out[0, j, i, t] = ds
                        out[1, j, i, t] = u_over_s
                        out[4, j, i, t] = dz / 2.0
                        out[3, j, i, t] = 0.0
                        if mode == 0:
                            out[5, j, i, t] = 0.0
                        elif mode == 1:
                            out[5, j, i, t] = u_over_s / 2.0
                        else:
                            out[5, j, i, t] = u_over_s
                    elif component == 1:
                        if mode == 1:
                            out[1, j, i, t] -= u_over_s
                        else:
                            out[1, j, i, t] -= 2.0 * u_over_s
                        out[3, j, i, t] = dz / 2.0
                        out[5, j, i, t] += (ds - u_over_s) / 2.0
                    else:
                        out[2, j, i, t] = dz
                        out[4, j, i, t] += ds / 2.0
                        if mode == 1:
                            out[3, j, i, t] += u_over_s / 2.0
                        elif mode == 2:
                            out[3, j, i, t] += u_over_s


def _nodes_array(nodes: np.ndarray, element_type: int) -> np.ndarray:
    """Validate and normalize element corner coordinates.

    Parameters
    ----------
    nodes : ndarray, shape (4, 2)
        Corner coordinates in metres, with columns (s, z).
    element_type : int
        Element mapping code from 0 through 3.

    Returns
    -------
    ndarray, shape (4, 2)
        Float64 C-contiguous corner array.
    """
    if element_type not in (0, 1, 2, 3):
        raise ValueError('element_type must be 0, 1, 2, or 3')
    nodes = np.ascontiguousarray(nodes, dtype=np.float64)
    if nodes.shape != (4, 2):
        raise ValueError('nodes must have shape (4, 2)')
    return nodes


def inside_element(s: float, z: float, nodes: np.ndarray,
                   element_type: int, tolerance: float
                   ) -> tuple[bool, float, float]:
    """Locate one cylindrical point in an element.

    Parameters
    ----------
    s : float
        Cylindrical horizontal coordinate in metres.
    z : float
        Cylindrical axial coordinate in metres.
    nodes : ndarray, shape (4, 2)
        Corner coordinates in metres, with columns (s, z).
    element_type : int
        Element mapping code from 0 through 3.
    tolerance : float
        Allowed excess beyond each local [-1, 1] bound.

    Returns
    -------
    inside : bool
        Whether the point falls inside the element within ``tolerance``.
    xi, eta : float
        Dimensionless local coordinates.
    """
    return _inside_element(s, z, _nodes_array(nodes, element_type),
                           element_type, tolerance)


def lagrange_interpol_2D_td(points1: np.ndarray, points2: np.ndarray,
                            coefficients: np.ndarray, x1: float,
                            x2: float) -> np.ndarray:
    """Interpolate a C-order wavefield at one local point.

    Parameters
    ----------
    points1 : ndarray, shape (nxi,)
        Local xi nodes.
    points2 : ndarray, shape (neta,)
        Local eta nodes.
    coefficients : ndarray, shape (neta, nxi, nt)
        C-order field values with time fastest.
    x1 : float
        Target xi coordinate.
    x2 : float
        Target eta coordinate.

    Returns
    -------
    ndarray, shape (nt,)
        Interpolated time series.
    """
    points1 = np.ascontiguousarray(points1, dtype=np.float64)
    points2 = np.ascontiguousarray(points2, dtype=np.float64)
    coefficients = np.ascontiguousarray(coefficients)
    if (coefficients.ndim != 3 or
            coefficients.shape[:2] != (len(points2), len(points1))):
        raise ValueError('coefficients must have shape (len(points2), len(points1), time)')
    return _interpolate_time_last(points1, points2, coefficients, x1, x2)


def find_theta(xi: np.ndarray, eta: np.ndarray, nodes: np.ndarray,
               element_type: int) -> np.ndarray:
    """Map every grid node to colatitude.

    Parameters
    ----------
    xi : ndarray, shape (nxi,)
        Local xi nodes.
    eta : ndarray, shape (neta,)
        Local eta nodes.
    nodes : ndarray, shape (4, 2)
        Corner coordinates in metres, with columns (s, z).
    element_type : int
        Element mapping code from 0 through 3.

    Returns
    -------
    ndarray, shape (neta, nxi)
        C-order colatitude grid in radians.
    """
    nodes = _nodes_array(nodes, element_type)
    xi = np.ascontiguousarray(xi, dtype=np.float64)
    eta = np.ascontiguousarray(eta, dtype=np.float64)
    out = np.empty((len(eta), len(xi)), dtype=np.float64, order='C')
    _find_theta(xi, eta, nodes, element_type, out)
    return out


def strain_td(u: np.ndarray, G: np.ndarray, GT: np.ndarray,
              xi: np.ndarray, eta: np.ndarray, npol: int, nsamp: int,
              nodes: np.ndarray, element_type: int, axial: bool,
              stype: str = 'monopole') -> np.ndarray:
    """Compute modal strain on one element for all time samples.

    Parameters
    ----------
    u : ndarray, shape (3, neta, nxi, nt)
        C-order displacement with components (s, phi, z) and time fastest.
    G : ndarray, shape (n, n)
        Eta derivative matrix.
    GT : ndarray, shape (n, n)
        Xi derivative matrix; use GLJ at the axis.
    xi : ndarray, shape (nxi,)
        Local xi nodes.
    eta : ndarray, shape (neta,)
        Local eta nodes.
    npol : int
        Polynomial degree, with n = npol + 1.
    nsamp : int
        Number of time samples, nt.
    nodes : ndarray, shape (4, 2)
        Corner coordinates in metres, with columns (s, z).
    element_type : int
        Element mapping code from 0 through 3.
    axial : bool
        Whether the element touches s = 0.
    stype : {'monopole', 'dipole', 'quadpole'}
        Azimuthal source type.

    Returns
    -------
    ndarray, shape (6, neta, nxi, nt)
        C-order strain in (ss, pp, zz, pz, sz, sp) order.
    """
    nodes = _nodes_array(nodes, element_type)
    modes = {'monopole': 0, 'dipole': 1, 'quadpole': 2}
    if stype not in modes:
        raise ValueError('unknown source type: ' + stype)
    u = np.ascontiguousarray(u, dtype=np.float64)
    G = np.ascontiguousarray(G, dtype=np.float64)
    GT = np.ascontiguousarray(GT, dtype=np.float64)
    xi = np.ascontiguousarray(xi, dtype=np.float64)
    eta = np.ascontiguousarray(eta, dtype=np.float64)
    n = npol + 1
    if (u.shape != (3, n, n, nsamp) or G.shape != (n, n) or
            GT.shape != (n, n) or xi.shape != (n,) or eta.shape != (n,)):
        raise ValueError('strain arrays do not match npol and nsamp')
    out = np.empty((6, n, n, nsamp), dtype=np.float64, order='C')
    _strain_time_last(u, G, GT, xi, eta, nodes, element_type, axial,
                      modes[stype], out)
    return out


__all__ = [
    'find_theta',
    'inside_element',
    'lagrange_interpol_2D_td',
    'strain_td',
]
