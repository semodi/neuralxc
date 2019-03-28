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

from neuralxc.projector.spher_grad import rlylm, grlylm

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
@pytest.mark.spher
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

    coords = np.random.rand(10,3)
    lmax = 10
    M = xc.projector.M_make_complex(lmax + 1)

    for c in coords:
        r, theta, phi = to_spher(c)

        ang_scipy = np.array([r**l*projector.angulars(l,m,theta, phi) \
                     for l in range(lmax + 1) \
                     for m in range(-l,l+1)])

        ang_soler = rlylm(lmax, c)
        ang_soler = M.dot(ang_soler)
        # print(ang_scipy)
        # print(ang_soler)
        assert np.allclose(ang_scipy, ang_soler)

    incr = 0.0000001
    inc_idx = 0
    for c in coords:
        dr = np.zeros(3)
        dr[inc_idx] += incr
        rp, thetap, phip = to_spher(c + dr)
        rm, thetam, phim = to_spher(c - dr)
        angp = np.array([rp**l*projector.angulars(l,m,thetap, phip) \
                     for l in range(lmax + 1) \
                     for m in range(-l,l+1)])

        angm = np.array([rm**l*projector.angulars(l,m,thetam, phim) \
                     for l in range(lmax + 1) \
                     for m in range(-l,l+1)])

        ang_solerp = rlylm(lmax, c + dr)
        ang_solerm = rlylm(lmax, c - dr)
        dang_fd_soler = (ang_solerp - ang_solerm)/(2*incr)
        dang_fd = (angp-angm)/(2*incr)

        dang_exact = grlylm(lmax, c)[inc_idx]
        dang_exact = M.dot(dang_exact)
        dang_fd_soler = M.dot(dang_fd_soler)
        assert np.allclose(dang_fd, dang_exact, atol=incr)
        assert np.allclose(dang_fd_soler, dang_exact, atol=incr)


@pytest.mark.skipif(not ase_found, reason='requires ase')
@pytest.mark.force
@pytest.mark.fast
def test_force_correction():

    benzene_nxc = xc.NeuralXC(os.path.join(test_dir,'benzene_test','benzene'))
    benzene_traj = ase.io.read(os.path.join(test_dir,'benzene_test','benzene.xyz'),'0')
    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir,
                                'benzene_test','benzene.RHOXC'))

    positions = benzene_traj.get_positions()/Bohr

    for incr_atom in [0]:
        incr = 0.001
        incr_idx = 1
        pp = np.array(positions)
        pm = np.array(pp)
        pp[incr_atom, incr_idx] += incr
        pm[incr_atom, incr_idx] -= incr

        species = benzene_traj.get_chemical_symbols()
        V, forces = benzene_nxc.get_V(rho, unitcell, grid, positions, species, calc_forces = True)[1]
        V = V/Hartree
        forces = forces/Hartree*Bohr
        Vp = benzene_nxc.get_V(rho, unitcell, grid, pp, species)[1]
        Vm = benzene_nxc.get_V(rho, unitcell, grid, pm, species)[1]
        dv = (unitcell[0,0]/grid[0])**3

        fp = dv*np.sum(Vp*rho)/Hartree*Bohr
        fm = dv*np.sum(Vm*rho)/Hartree*Bohr

        print((fp-fm)/(2*incr))
        print(forces[incr_atom,incr_idx])
    assert False
