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
    for inc_idx in range(3):
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

    incr = 0.00001
    l = 6
    m = -3
    for inc_idx in range(3):
        pick = sum([1 for l_ in range(l) for m_ in range(-l_,l_+1)])
        pick += (l+m)
        dr = np.zeros(3)
        dr[inc_idx] += incr
        mesh = np.meshgrid(*[np.linspace(0,2,6)]*3)
        rp, thetap, phip = to_spher([r + dr_ for r,dr_ in zip(mesh,dr)])
        rm, thetam, phim = to_spher([r - dr_ for r,dr_ in zip(mesh,dr)])

        angp = rp**l*projector.angulars(l,m,thetap, phip)

        angm = rm**l*projector.angulars(l,m,thetam, phim)

        dang_fd = (angp-angm)/(2*incr)
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
        dangs = dangs.reshape(len(dangs),*X.shape,3)[pick,:,:,:,inc_idx]
        assert np.allclose(dangs, dang_fd, atol = incr)
        
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
        forces = forces
        Vp = benzene_nxc.get_V(rho, unitcell, grid, pp, species)[1]
        Vm = benzene_nxc.get_V(rho, unitcell, grid, pm, species)[1]
        fig = xc.utils.visualize.plot_density_cut(Vp-Vm, rmax = [40]*3)
        fig.savefig('./vpvm.pdf')
        fig = xc.utils.visualize.plot_density_cut(Vp, rmax = [40]*3)
        fig.savefig('./vp.pdf')

        dv = (unitcell[0,0]/grid[0])**3
        fp = dv*np.sum(Vp*rho)
        fm = dv*np.sum(Vm*rho)

        forces_fd = (fp-fm)/(2*incr)
        print(forces_fd)
        print(forces[incr_atom,incr_idx])
        # assert np.allclose(forces_fd, forces[incr_atom, incr_idx])

@pytest.mark.delbasis
def test_del_basis():
    benzene_traj = ase.io.read(os.path.join(test_dir,'benzene_test','benzene.xyz'),'0')
    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir,
                                'benzene_test','benzene.RHOXC'))
    basis_set = {
                'C': {'n' : 4, 'l' : 3, 'r_o': 1},
                'H': {'n' : 4, 'l' : 2, 'r_o': 1.5}
                }
    positions = benzene_traj.get_positions()/Bohr
    species = benzene_traj.get_chemical_symbols()

    projector = xc.projector.DensityProjector(unitcell, grid, basis_set)
    n = 0
    l = 0
    m = 0
    for incr_atom in [0]:
        incr = 0.001
        incr_idx = 1
        pp = np.array(positions)
        pm = np.array(pp)
        pp[incr_atom, incr_idx] += incr
        pm[incr_atom, incr_idx] -= incr


        basis_rep_p = projector.get_basis_rep_dict(rho, pp, species)['C'][0]['{},{},{}'.format(n,l,m)]
        basis_rep_m = projector.get_basis_rep_dict(rho, pm, species)['C'][0]['{},{},{}'.format(n,l,m)]
        print(basis_rep_p)
        print(basis_rep_m)
        print((basis_rep_p-basis_rep_m)/(2*incr))

        box = projector.box_around(positions[0],1)
        print(type(projector.W['C']))
        dbasis = del_basis(projector, rho, box, n, l, m, 1, projector.W['C'])
        print(dbasis[incr_idx])
        assert False

def del_basis(self, rho, box, n, l,m, r_o, W = None):
    R, Theta, Phi = box['radial']
    Xm, Ym, Zm = box['mesh']
    X, Y, Z = box['real']
    n_l = l +1
    #Build angular part of basis functions
    ang = self.angulars(l, m, Theta, Phi)

    # Derivatives of spherical harmonic
    M = xc.projector.M_make_complex(n_l)
    dangs = []
    for il in range(n_l**2):
        dangs.append(np.zeros([len(X.flatten()),3]))

    for ir, r in enumerate(zip(X.flatten(),Y.flatten(),Z.flatten())):
        vecspher = xc.projector.spher_grad.grlylm(n_l - 1, r) # shape: (3, n_l*n_l)
        for il, vs in enumerate(vecspher.T):
            dangs[il][ir] = vs

    dangs = np.einsum('ij,jkl -> ikl', M, np.array(dangs))

    pick = sum([1 for l_ in range(l) for m_ in range(-l_,l_+1)])
    pick += (l+m)
    print(pick)
    dangs = dangs.reshape(len(dangs),*X.shape,3)
    dang = dangs[pick]

    #Build radial part of b.f.
    if not isinstance(W, np.ndarray):
        W = self.get_W(r_o, n_rad) # Matrix to orthogonalize radial basis

    drads = self.dradials(R, r_o, W)[n]
    rads = self.radials(R, r_o, W)[n]
    radsr = np.array(rads)
    # radsr[R==0] = 0
    radsr = radsr/R

    rhat = [X/R, Y/R, Z/R]
    rho = rho[box['mesh']]
    print(np.linalg.norm(drads))
    print(np.linalg.norm(dang[:,:,:,:]))
    return [np.sum(rho*(ang * (drads - l*radsr) * rhat[ix]) + \
                     (rads/(R**l)*dang[:,:,:,ix]))*self.V_cell for ix in range(3)]
