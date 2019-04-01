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

    # Check spherical harmonics
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

    # Check gradients of spherical harmonics
    incr = 0.00001
    for inc_idx in range(3):
        dr = np.zeros(3)
        dr[inc_idx] += incr
        mesh = np.meshgrid(*[np.linspace(0,2,6)]*3)
        rp, thetap, phip = to_spher([r + dr_ for r,dr_ in zip(mesh,dr)])
        rm, thetam, phim = to_spher([r - dr_ for r,dr_ in zip(mesh,dr)])

        angp = [rp**l*projector.angulars(l,m,thetap, phip) \
                     for l in range(lmax + 1) \
                     for m in range(-l,l+1)]

        angm = [rm**l*projector.angulars(l,m,thetam, phim) \
                     for l in range(lmax + 1) \
                     for m in range(-l,l+1)]

        dang_fd = [(p-m)/(2*incr) for p,m in zip(angp,angm)]
        n_l = lmax
        M = xc.projector.M_make_complex(n_l)
        X, Y, Z = mesh
        dangs = []
        for il in range(n_l**2):
            dangs.append(np.zeros([len(X.flatten()),3]))

        for ir, r in enumerate(zip(X.flatten(),Y.flatten(),Z.flatten())):
            print(r)
            vecspher = xc.projector.spher_grad.grlylm(n_l - 1, r) # shape: (3, n_l*n_l)
            for il, vs in enumerate(vecspher.T):
                dangs[il][ir] = vs

        dangs = np.einsum('ij,jkl -> ikl', M, np.array(dangs))
        dangs = dangs.reshape(len(dangs),*X.shape,3)[:,:,:,:,inc_idx]
        for idx, (exact, fd) in enumerate(zip(dangs, dang_fd)):
            assert np.allclose(exact, fd, atol = incr)


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
    species = benzene_traj.get_chemical_symbols()

    def get_V_shifted(self, rho, unitcell, grid, positions, positions_shifted, species, calc_forces= False):
        """ only defined to calculate basis set contribution to forces with numerical derivatives.
        For finite difference, the descriptors should be calculated using the original positions,
        whereas V will then be built with displaced atoms
        """
        projector = xc.projector.DensityProjector(unitcell, grid,
            self._pipeline.get_basis_instructions())

        symmetrize_dict = {'basis': self._pipeline.get_basis_instructions()}
        symmetrize_dict.update(self._pipeline.get_symmetrize_instructions())

        symmetrizer = xc.symmetrizer.symmetrizer_factory(symmetrize_dict)

        C = projector.get_basis_rep(rho, positions, species)

        D = symmetrizer.get_symmetrized(C)
        dEdC = symmetrizer.get_gradient(self._pipeline.get_gradient(D))
        E = self._pipeline.predict(D)[0]

        return E, projector.get_V(dEdC, positions_shifted, species, calc_forces, rho)

    V, forces = benzene_nxc.get_V(rho, unitcell, grid, positions, species, calc_forces = True)[1]

    assert np.allclose(np.sum(forces, axis = 0), np.zeros(3), atol = 1e-6)
    for incr_atom in [0,1,2,3]:
        for incr_dx in range(3):
            incr = 0.00001
            incr_idx = 1
            pp = np.array(positions)
            pm = np.array(pp)
            pp[incr_atom, incr_idx] += incr
            pm[incr_atom, incr_idx] -= incr

            Vp = get_V_shifted(benzene_nxc, rho, unitcell, grid, positions, pp, species)[1]
            Vm = get_V_shifted(benzene_nxc, rho, unitcell, grid, positions,pm, species)[1]

            dv = (unitcell[0,0]/grid[0])**3
            fp = dv*np.sum(Vp*rho)
            fm = dv*np.sum(Vm*rho)

            forces_fd = (fp-fm)/(2*incr)
            print(forces_fd)
            print(forces[incr_atom,incr_idx])
            assert np.allclose(-forces_fd, forces[incr_atom, incr_idx],atol = incr)
