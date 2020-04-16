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
import periodictable
from time import time
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

test_dir = os.path.dirname(os.path.abspath(__file__))

save_test_density_projector = False
save_siesta_density_getter = False
save_test_symmetrizer = False
save_grouped_transformer = False



@pytest.mark.torch
def test_torch_projector():
    benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
    benzene_traj = ase.io.read(os.path.join(test_dir, 'benzene_test', 'benzene.xyz'), '0')
    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))
    positions = benzene_traj.get_positions() / Bohr
    species = benzene_traj.get_chemical_symbols()
    a = np.linalg.norm(unitcell, axis=1) / grid[:3]

    benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
    start = time()
    C = benzene_nxc.projector.get_basis_rep(rho, positions, species)
    for spec in C:
        C[spec] *= 0
    C['C'][0,0] = 1
    psi_numpy = benzene_nxc.projector.get_V(C, positions, species)
    C_numpy = tuple(benzene_nxc.projector.get_basis_rep(rho, positions, species).values())
    benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
    benzene_nxc._pipeline.to_torch()
    benzene_nxc._pipeline.basis_instructions['projector_type'] = 'ortho_torch'
    benzene_nxc._pipeline.symmetrize_instructions['symmetrizer_type'] = 'casimir_torch'
    benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)

    rho = torch.from_numpy(rho).double()
    unitcell = torch.from_numpy(unitcell).double()
    unitcell.requires_grad = False
    grid = torch.from_numpy(grid).double()
    positions = torch.from_numpy(positions).double()
    species = torch.Tensor([getattr(periodictable,s).number for s in species])
    a = torch.from_numpy(a).double()

    class Module(torch.nn.Module):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.projector = benzene_nxc.projector

        def forward(self, rho, positions, species, unitcell, grid, a):
             x = self.projector(rho, positions, species, unitcell, grid, a)
             return tuple(x.values())


    some_input = unitcell
    projector = Module()
    C_torch = projector(rho, positions, species, unitcell, grid, a)
    with torch.jit.optimized_execution(should_optimize=True):
        compiled = torch.jit.trace(projector, (rho, positions, species, unitcell, grid, a), optimize=True, check_trace = False)
        C_compiled = compiled(rho, positions, species, unitcell, grid, a)

    for ct, cn, cc in zip(C_torch, C_numpy, C_compiled):
        assert np.allclose(ct,cn)
        assert np.allclose(ct,cc)

    rho.requires_grad = True
    C_torch = projector(rho, positions, species, unitcell, grid, a)
    C_torch[0][0][0].backward()
    psi_torch = (rho.grad/benzene_nxc.projector.V_cell).detach().numpy()

    with torch.jit.optimized_execution(should_optimize=True):
        rho.grad.zero_()
        C_torch_comp = compiled(rho, positions, species, unitcell, grid, a)
        C_torch_comp[0][0][0].backward()
        psi_compiled = (rho.grad/benzene_nxc.projector.V_cell).detach().numpy()

    assert np.allclose(psi_numpy, psi_torch)
    assert np.allclose(psi_torch, psi_compiled)


@pytest.mark.skipif(not ase_found, reason='requires ase')
@pytest.mark.torch
def test_neuralxc_benzene_torch():

    benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
    benzene_traj = ase.io.read(os.path.join(test_dir, 'benzene_test', 'benzene.xyz'), '0')
    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))

    a = np.linalg.norm(unitcell, axis=1) / grid[:3]
    positions = benzene_traj.get_positions() / Bohr
    species = benzene_traj.get_chemical_symbols()
    benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)

    start = time()
    for i in range(1):
        V_classical = benzene_nxc.get_V(rho)
    end = time()
    normal_time = (-start + end)/1

    benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
    benzene_nxc._pipeline.to_torch()
    benzene_nxc._pipeline.basis_instructions['projector_type'] = 'ortho_torch'
    benzene_nxc._pipeline.symmetrize_instructions['symmetrizer_type'] = 'casimir_torch'
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))
    benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)

    rho = torch.from_numpy(rho).double()
    rho.requires_grad = True
    unitcell = torch.from_numpy(unitcell).double()
    grid = torch.from_numpy(grid).double()
    positions = torch.from_numpy(positions).double()
    species = torch.Tensor([getattr(periodictable,s).number for s in species])
    a = torch.from_numpy(a).double()

    class E_predictor(torch.nn.Module):

        def __init__(self):
            torch.nn.Module.__init__(self)

        def forward(self, rho, positions, species, unitcell, grid, a):
             x = benzene_nxc.symmetrizer(benzene_nxc.projector(rho, positions, species, unitcell, grid, a))
             for steps in benzene_nxc._pipeline.steps[:]:
                 x = steps[1].forward(x)
             return x

    epred = E_predictor()
    E = epred(rho, positions, species, unitcell, grid, a)
    E.backward()
    assert np.allclose(V_classical[1],(rho.grad/benzene_nxc.projector.V_cell).detach().numpy())
    with torch.jit.optimized_execution(should_optimize=True):
        compiled_model = torch.jit.trace(epred, (rho, positions, species, unitcell, grid, a), check_trace = False)
        rho.grad.zero_()
        E_compiled = compiled_model(rho, positions, species, unitcell, grid, a)
        E_compiled.backward()
    assert np.allclose(V_classical[1],(rho.grad/benzene_nxc.projector.V_cell).detach().numpy())
