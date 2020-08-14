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
import dill as pickle
import copy
import matplotlib.pyplot as plt
from neuralxc.constants import Bohr, Hartree
try:
    import ase
    ase_found = True
except ModuleNotFoundError:
    ase_found = False
try:
    import torch
    torch_found =True
except ModuleNotFoundError:
    torch_found = False
# try:
#     import pyscf
#     pyscf_found = True
# except ModuleNotFoundError:
#     pyscf_found = False
pyscf_found = False
#
test_dir = os.path.dirname(os.path.abspath(__file__))

@pytest.mark.skipif(not pyscf_found, reason='requires pyscf')
@pytest.mark.gaussian
def test_radial_gaussian():
    from pyscf import gto, dft
    mol = gto.M(atom='O  0  0  0; H  0 1 0 ; H 0 0 1', basis='6-31g')
    mf = dft.RKS(mol)
    # mf.xc = 'LDA'
    mf.grids.level = 5
    mf.kernel()

    basis_instructions = { "application":"pyscf",
                            "spec_agnostic": True,
                            "basis":os.path.join(test_dir, "basis-test")}

    projector = xc.projector.DensityProjector(mol=mol, basis_instructions=basis_instructions)
    coeff_analytical = projector.get_basis_rep(mf.make_rdm1())

    basis_instructions = {"application":"pyscf_rad",
                            "spec_agnostic": True,
                            "projector_type": "gaussian_radial",
                            "X": {
                                    "basis": os.path.join(test_dir, "basis-test"),
                                    "sigma": 20
                            }}

    rho = pyscf.dft.numint.get_rho(mf._numint, mol, mf.make_rdm1(), mf.grids)

    projector = xc.projector.DensityProjector(basis_instructions = basis_instructions,
         grid_coords = mf.grids.coords, grid_weights = mf.grids.weights)
    coeff_grid = projector.get_basis_rep(rho,  np.array([[0,0,0],[0,1,0],[0,0,1]])/Bohr,
        ['X','X','X'])

    assert np.allclose(np.linalg.norm(coeff_analytical['X']), np.linalg.norm(coeff_grid['X']))

@pytest.mark.gaussian
@pytest.mark.parametrize('torch',['','_torch'])
def test_gaussian_projector(torch):
    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'h2o.RHO'))

    basis_instructions = {"application":"siesta",
                            "spec_agnostic": True,
                            "projector_type": "gaussian" + torch,
                            "X": {
                                    "basis": os.path.join(test_dir, "basis-test"),
                                    "sigma": 2
                            }}

    density_projector = xc.projector.DensityProjector(unitcell=unitcell, grid=grid, basis_instructions=basis_instructions)

    positions = np.array([[0.0, 0.0, 0.0], [-0.75846035, -0.59257417, 0.0], [0.75846035, -0.59257417, 0.0]
                          ]) / xc.constants.Bohr

    basis_rep = density_projector.get_basis_rep(rho, positions=positions, species=['X', 'X', 'X'])

    with open(os.path.join(test_dir, 'h2o_gaussian_rep.pckl'), 'rb') as file:
        ref = pickle.load(file)
    for spec in basis_rep:
        assert np.allclose(basis_rep[spec], ref[spec])

@pytest.mark.gaussian
def test_gaussian_compiled():
    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'h2o.RHO'))

    basis_instructions = {"application":"siesta",
                            "spec_agnostic": True,
                            "projector_type": "gaussian_torch",
                            "X": {
                                    "basis": os.path.join(test_dir, "basis-test"),
                                    "sigma": 2
                            }}

    density_projector = xc.projector.DensityProjector(unitcell=unitcell, grid=grid, basis_instructions=basis_instructions)

    basis_models, projector_models = xc.ml.network.compile_projector(density_projector)

    my_box = torch.Tensor([[0, grid[i]] for i in range(3)])
    unitcell = torch.from_numpy(unitcell).double()
    grid = torch.from_numpy(grid).double()
    positions = torch.from_numpy(np.array([[0.0, 0.0, 0.0], [-0.75846035, -0.59257417, 0.0], [0.75846035, -0.59257417, 0.0]
                          ]) / xc.constants.Bohr)
    rho = torch.from_numpy(rho)
    coeffs = []
    for pos in positions:
        rad,ang,box = basis_models['X'](pos, unitcell, grid, my_box)
        coeffs.append(projector_models['X'](rho, pos, unitcell, grid, rad, ang, box).detach().numpy())

    basis_rep ={'X': np.array(coeffs)}

    with open(os.path.join(test_dir, 'h2o_gaussian_rep.pckl'), 'rb') as file:
        ref = pickle.load(file)
    for spec in basis_rep:
        assert np.allclose(basis_rep[spec], ref[spec])




    #
    # basis_rep = density_projector.get_basis_rep(rho, positions=positions, species=['X', 'X', 'X'])
    #
    # with open(os.path.join(test_dir, 'h2o_gaussian_rep.pckl'), 'rb') as file:
    #     ref = pickle.load(file)
    # for spec in basis_rep:
    #     assert np.allclose(basis_rep[spec], ref[spec])
