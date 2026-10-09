#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrappers around some functions from the finite_elem_mapping module from
AxiSEM's kernel module.

:copyright:
    Lion Krischer (lion.krischer@gmail.com), 2020
:license:
    GNU Lesser General Public License, Version 3 [non-commercial/academic use]
    (http://www.gnu.org/copyleft/lgpl.html)

: Modified by nqdu on Dec 12th, 2023
"""
from lib import libsem
import numpy as np

def inside_element(s, z, nodes, element_type, tolerance):
    nodes = np.require(nodes, requirements=["F_CONTIGUOUS"])
    isin,xi,eta = libsem.inside_element_warp(s,z,nodes,element_type,tolerance)

    return isin, xi, eta

def lagrange_interpol_2D_td(points1, points2, coefficients, x1, x2):  # NOQA
    points1 = np.require(
        points1, dtype=np.float64, requirements=["F_CONTIGUOUS"]
    )
    points2 = np.require(
        points2, dtype=np.float64, requirements=["F_CONTIGUOUS"]
    )
    coefficients = np.require(
        coefficients, dtype=np.float64, requirements=["F_CONTIGUOUS"]
    )


    # Should be safe enough. This was never raised while extracting a lot of
    # seismograms.
    assert len(points1) == len(points2)

    out = libsem.lagrange_2D(points1,points2,coefficients,x1,x2)

    return out

def strain_td(u, G, GT, xi, eta, npol, nsamp, nodes, element_type, axial,stype="monopole"):

    assert(stype in ["monopole","dipole","quadpole"])

    u = np.require(u, dtype=np.float64, requirements=["F_CONTIGUOUS"])
    G = np.require(G, dtype=np.float64, requirements=["F_CONTIGUOUS"])  # NOQA
    GT = np.require(
        GT, dtype=np.float64, requirements=["F_CONTIGUOUS"]  # NOQA
    )
    xi = np.require(xi, dtype=np.float64, requirements=["F_CONTIGUOUS"])
    eta = np.require(eta, dtype=np.float64, requirements=["F_CONTIGUOUS"])
    nodes = np.require(nodes, dtype=np.float64, requirements=["F_CONTIGUOUS"])

    strain = libsem.strain_td_warp(u,G,GT,xi,eta,nodes,element_type,axial,stype)
    strain = np.reshape(strain,(nsamp, npol + 1, npol + 1, 6),order='F')

    return strain 

def find_theta(xi,eta,nodes,element_type) -> np.ndarray :
    xi = np.require(xi, dtype=np.float64, requirements=["F_CONTIGUOUS"])
    eta = np.require(eta, dtype=np.float64, requirements=["F_CONTIGUOUS"])
    nodes = np.require(nodes, dtype=np.float64, requirements=["F_CONTIGUOUS"])

    return libsem.find_theta(xi,eta,nodes,element_type)