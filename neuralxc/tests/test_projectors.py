"""
Unit and regression test for the neuralxc package.
"""

import copy
import os
import sys
from abc import ABC, abstractmethod

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Import package, test suite, and other packages as needed
import neuralxc as xc
from neuralxc.constants import Bohr, Hartree

try:
    import ase
    ase_found = True
except ModuleNotFoundError:
    ase_found = False
try:
    import torch
    torch_found = True
except ModuleNotFoundError:
    torch_found = False
try:
    import pyscf
    pyscf_found = True
except ModuleNotFoundError:
    pyscf_found = False
#
test_dir = os.path.dirname(os.path.abspath(__file__))

save_test_density_projector = False
save_test_radial_projector = False


@pytest.mark.fast
@pytest.mark.project
@pytest.mark.parametrize('projector_type', ['ortho'])
def test_density_projector(projector_type):

    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'h2o.RHO'))

    basis_set = {'O': {'n': 2, 'l': 3, 'r_o': 1}, 'H': {'n': 2, 'l': 2, 'r_o': 1.5}, 'projector_type': projector_type}

    density_projector = xc.projector.DensityProjector(unitcell=unitcell, grid=grid, basis_instructions=basis_set)

    positions = np.array([[0.0, 0.0, 0.0], [-0.75846035, -0.59257417, 0.0], [0.75846035, -0.59257417, 0.0]
                          ]) / xc.constants.Bohr

    basis_rep = density_projector.get_basis_rep(rho, positions=positions, species=['O', 'H', 'H'])

    if 'ortho' in projector_type:
        if save_test_density_projector:
            with open(os.path.join(test_dir, 'h2o_rep.pckl'), 'wb') as file:
                pickle.dump(basis_rep, file)
        else:
            with open(os.path.join(test_dir, 'h2o_rep.pckl'), 'rb') as file:
                basis_rep_ref = pickle.load(file)

            for spec in basis_rep:
                assert np.allclose(basis_rep[spec], basis_rep_ref[spec])


@pytest.mark.fast
@pytest.mark.project
@pytest.mark.parametrize('grid_type', ['', '_radial'])
@pytest.mark.parametrize('rad_type', ['ortho', 'gaussian'])
def test_jacobs_projector(rad_type, grid_type):

    projector_type = rad_type + grid_type
    positions = np.array([[0.0, 0.0, 0.0], [-0.75846035, -0.59257417, 0.0], [0.75846035, -0.59257417, 0.0]
                          ]) / xc.constants.Bohr

    if rad_type == 'ortho':
        basis_instructions = {'X': {'n': 2, 'l': 3, 'r_o': 1}, 'projector_type': projector_type, 'grad': 1}
    else:
        basis_instructions = {
            "application": "siesta",
            "spec_agnostic": True,
            "projector_type": projector_type,
            "X": {
                "basis": os.path.join(test_dir, "basis-test"),
                "sigma": 2
            },
            'grad': 1
        }

    if grid_type == '_radial':
        from pyscf import dft, gto
        mol = gto.M(atom='O  0  0  0; H  0 1 0 ; H 0 0 1', basis='6-31g*')
        mf = dft.RKS(mol)
        mf.xc = 'PBE'
        mf.grids.level = 5
        mf.kernel()
        rho = pyscf.dft.numint.get_rho(mf._numint, mol, mf.make_rdm1(), mf.grids)
        print('Rho shape', rho.shape)
        print('Weights shape', mf.grids.weights.shape)
        density_projector = xc.projector.DensityProjector(grid_coords=mf.grids.coords,
                                                          grid_weights=mf.grids.weights,
                                                          basis_instructions=basis_instructions)
    else:
        density_getter = xc.utils.SiestaDensityGetter(binary=True)
        rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'h2o.RHO'))
        density_projector = xc.projector.DensityProjector(unitcell=unitcell,
                                                          grid=grid,
                                                          basis_instructions=basis_instructions)

    rho = np.stack([rho, rho])

    basis_rep = density_projector.get_basis_rep(rho, positions=positions, species=['X', 'X', 'X'])
    for key, val in basis_rep.items():
        l = val.shape[-1] // 2
        assert np.allclose(val[..., :l], val[..., l:])

    if rad_type == 'ortho':
        symmetrize_instructions = {'symmetrizer_type': 'trace', 'basis': basis_instructions}
        sym = xc.symmetrizer.Symmetrizer(symmetrize_instructions)
        D = sym.get_symmetrized(basis_rep)['X']
        l = D.shape[-1] // 2
        assert np.allclose(D[:, :l], D[:, l:])


@pytest.mark.fast
@pytest.mark.radial
@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
@pytest.mark.parametrize('projector_type', ['ortho_radial'])
def test_radial_projector(projector_type):
    from pyscf import dft, gto
    mol = gto.M(atom='O  0  0  0; H  0 1 0 ; H 0 0 1', basis='6-31g*')
    mf = dft.RKS(mol)
    mf.xc = 'PBE'
    mf.grids.level = 5
    mf.kernel()

    rho = pyscf.dft.numint.get_rho(mf._numint, mol, mf.make_rdm1(), mf.grids)

    basis_set = {'O': {'n': 2, 'l': 3, 'r_o': 1}, 'H': {'n': 2, 'l': 2, 'r_o': 1.5}, 'projector_type': projector_type}

    density_projector = xc.projector.DensityProjector(grid_coords=mf.grids.coords,
                                                      grid_weights=mf.grids.weights,
                                                      basis_instructions=basis_set)

    positions = np.array([[0.0, 0.0, 0.0], [0, 1, 0.0], [0, 0, 1]]) / xc.constants.Bohr

    basis_rep = density_projector.get_basis_rep(rho, positions=positions, species=['O', 'H', 'H'])

    if projector_type in ['ortho_radial']:
        if save_test_radial_projector:
            with open(os.path.join(test_dir, 'h2o_rad.pckl'), 'wb') as file:
                pickle.dump(basis_rep, file)
    with open(os.path.join(test_dir, 'h2o_rad.pckl'), 'rb') as file:
        basis_rep_ref = pickle.load(file)

    for spec in basis_rep:
        assert np.allclose(basis_rep[spec], basis_rep_ref[spec])


@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
@pytest.mark.gaussian
def test_radial_gaussian():
    from pyscf import dft, gto
    mol = gto.M(atom='O  0  0  0; H  0 1 0 ; H 0 0 1', basis='6-31g')
    mf = dft.RKS(mol)
    # mf.xc = 'LDA'
    mf.grids.level = 5
    mf.kernel()

    basis_instructions = {"application": "pyscf", "spec_agnostic": True, "basis": os.path.join(test_dir, "basis-test")}

    projector = xc.projector.DensityProjector(mol=mol, basis_instructions=basis_instructions)
    coeff_analytical = projector.get_basis_rep(mf.make_rdm1())

    basis_instructions = {
        "application": "pyscf_rad",
        "spec_agnostic": True,
        "projector_type": "gaussian_radial",
        "X": {
            "basis": os.path.join(test_dir, "basis-test"),
            "sigma": 20
        }
    }

    rho = pyscf.dft.numint.get_rho(mf._numint, mol, mf.make_rdm1(), mf.grids)

    projector = xc.projector.DensityProjector(basis_instructions=basis_instructions,
                                              grid_coords=mf.grids.coords,
                                              grid_weights=mf.grids.weights)
    coeff_grid = projector.get_basis_rep(rho, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]) / Bohr, ['X', 'X', 'X'])
    assert np.allclose(np.linalg.norm(coeff_analytical['X']), np.linalg.norm(coeff_grid['X']))


@pytest.mark.gaussian
def test_gaussian_projector(torch=''):
    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'h2o.RHO'))

    basis_instructions = {
        "application": "siesta",
        "spec_agnostic": True,
        "projector_type": "gaussian" + torch,
        "X": {
            "basis": os.path.join(test_dir, "basis-test"),
            "sigma": 2
        }
    }

    density_projector = xc.projector.DensityProjector(unitcell=unitcell,
                                                      grid=grid,
                                                      basis_instructions=basis_instructions)

    positions = np.array([[0.0, 0.0, 0.0], [-0.75846035, -0.59257417, 0.0], [0.75846035, -0.59257417, 0.0]
                          ]) / xc.constants.Bohr

    basis_rep = density_projector.get_basis_rep(rho, positions=positions, species=['X', 'X', 'X'])

    with open(os.path.join(test_dir, 'h2o_gaussian_rep.pckl'), 'rb') as file:
        ref = pickle.load(file)
    for spec in basis_rep:
        assert np.allclose(basis_rep[spec], ref[spec])


@pytest.mark.gaussian
def test_gaussian_serialized():
    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'h2o.RHO'))

    basis_instructions = {
        "application": "siesta",
        "spec_agnostic": True,
        "projector_type": "gaussian",
        "X": {
            "basis": os.path.join(test_dir, "basis-test"),
            "sigma": 2
        }
    }

    density_projector = xc.projector.DensityProjector(unitcell=unitcell,
                                                      grid=grid,
                                                      basis_instructions=basis_instructions)

    basis_models, projector_models = xc.ml.pipeline.serialize_projector(density_projector)

    my_box = torch.Tensor([[0, grid[i]] for i in range(3)])
    unitcell = torch.from_numpy(unitcell).double()
    grid = torch.from_numpy(grid).double()
    positions = torch.from_numpy(
        np.array([[0.0, 0.0, 0.0], [-0.75846035, -0.59257417, 0.0], [0.75846035, -0.59257417, 0.0]]) /
        xc.constants.Bohr)
    rho = torch.from_numpy(rho)
    coeffs = []
    for pos in positions:
        rad, ang, box = basis_models['X'](pos, unitcell, grid, my_box)
        coeffs.append(projector_models['X'](rho, pos, unitcell, grid, rad, ang, box).detach().numpy())

    basis_rep = {'X': np.array(coeffs)}

    with open(os.path.join(test_dir, 'h2o_gaussian_rep.pckl'), 'rb') as file:
        ref = pickle.load(file)
    for spec in basis_rep:
        assert np.allclose(basis_rep[spec], ref[spec])
