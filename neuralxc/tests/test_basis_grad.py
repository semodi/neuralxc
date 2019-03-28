"""
Unit and regression test for the neuralxc package.
"""

# Import package, test suite, and other packages as needed
import neuralxc as xc
import pytest
import sys
import numpy as np
import os
from neuralxc.doc_inherit import doc_inherit
from abc import ABC, abstractmethod
import pickle
import copy
import matplotlib.pyplot as plt
from neuralxc.constants import Bohr, Hartree
try:
    import ase
    ase_found = True
except ModuleNotFoundError:
    ase_found = False

from neuralxc.projector.spher_grad import rlylm

test_dir = os.path.dirname(os.path.abspath(__file__))

@pytest.mark.fast
@pytest.mark.parametrize('r_o',[1.21, 2.34])
@pytest.mark.parametrize('n',[1, 4, 6])
def test_drad(r_o, n):

    density_getter = xc.utils.SiestaDensityGetter(binary = True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir,'h2o.RHO'))

    basis_set = {
                'O': {'n' : 2, 'l' : 3, 'r_o': 1},
                'H': {'n' : 2, 'l' : 2, 'r_o': 1.5}
                }

    projector = xc.projector.DensityProjector(unitcell, grid, basis_set)

    r = np.linspace(0, r_o, 100)
    incr = 0.000001
    rp = r + incr
    rm = r - incr
    for a in range(1,10,3):
        gp =  projector.g(rp, r_o, a)
        gm =  projector.g(rm, r_o, a)
        dg_fd = (gp-gm)/(2*incr)
        dg = projector.dg(r, r_o, a)
        assert np.allclose(dg_fd, dg, atol = incr)

    W = projector.get_W(r_o, n)
    radp = projector.radials(rp, r_o, W)
    radm = projector.radials(rm, r_o, W)
    drad_fd = [(p-m)/(2*incr) for p,m in zip(radp, radm)]
    drad = projector.dradials(r, r_o, W)

    for exact, fd in zip(drad, drad_fd):
        assert np.allclose(exact, fd, atol = incr)

@pytest.mark.fast
def test_dspher():

    def to_spher(p):
        x,y,z = p
        r = np.sqrt(x**2 + y**2 + z**2)

        phi = np.arctan2(y,x)
        theta = np.arccos(z/r, where = (r != 0))
        return r, theta, phi

    density_getter = xc.utils.SiestaDensityGetter(binary = True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir,'h2o.RHO'))

    basis_set = {
                'O': {'n' : 2, 'l' : 3, 'r_o': 1},
                'H': {'n' : 2, 'l' : 2, 'r_o': 1.5}
                }

    projector = xc.projector.DensityProjector(unitcell, grid, basis_set)

    coords = np.random.rand(5,3)
    incr = 0.000001
    inc_idx = 0
    l = 2
    m = 0

    pick = sum([1 for l_ in range(l+1) for m_ in range(-l_,m)])
    for c in coords:
        dr = np.zeros(3)
        dr[inc_idx] += incr
        rp, thetap, phip = to_spher(c + dr)
        rm, thetam, phim = to_spher(c - dr)
        angp = rp**l*projector.angulars(l,m,thetap, phip)
        angm = rm**l*projector.angulars(l,m,thetam, phim)
        dang_fd = (angp-angm)/(2*incr)

        dang_exact = rlylm(l, c)[inc_idx,pick]

        print(dang_exact)
        print(dang_fd)
        assert False
