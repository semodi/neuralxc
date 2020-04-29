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

# @pytest.mark.torchmbp
# def test_compile_mbpolmodel():
#
#     model = xc.NeuralXC(test_dir[:-len('neuralxc/tests/')] + '/examples/models/MB-pol/model')
#     xc.ml.network.compile_model(model, os.path.join(test_dir, 'mbp.nxc.jit')

@pytest.mark.torch
def test_mybox():

    benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
    benzene_traj = ase.io.read(os.path.join(test_dir, 'benzene_test', 'benzene.xyz'), '0')
    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))
    grid_np = np.array(grid)
    positions = benzene_traj.get_positions() / Bohr
    # Break symmetries
    positions[:,0] += 0.02
    positions[:,1] += 0.01
    positions[:,2] += 0.12
    species = benzene_traj.get_chemical_symbols()
    a = np.linalg.norm(unitcell, axis=1) / grid[:3]
    benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
    C = benzene_nxc.projector.get_basis_rep(rho, positions, species)
    C = {spec: C[spec].tolist() for spec in C}
    # my_box[0,1] = grid[0]/2
    basis = benzene_nxc._pipeline.get_basis_instructions()

    benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
    xc.ml.network.compile_model(benzene_nxc, 'benzene.nxc.jit',
        override=True)
    benzene_nxc = xc.neuralxc.NeuralXCJIT('benzene.nxc.jit')
    basis_models = benzene_nxc.basis_models
    projector_models = benzene_nxc.projector_models

    unitcell = torch.from_numpy(unitcell).double()
    grid = torch.from_numpy(grid).double()
    positions = torch.from_numpy(positions).double()
    a = torch.from_numpy(a).double()

    def is_in_box(pos, rc, my_box_d, grid_d, a): #Not necessary but can be used
                                                 #(might promise some speedup)
        return True
        grid = grid_d * a[0]
        my_box = my_box_d * a[0]
        posm = pos - rc
        posp = pos + rc
        posm = torch.fmod(torch.ceil(torch.abs(posm)/grid)*grid + posm,grid)
        posp = torch.fmod(torch.ceil(torch.abs(posp)/grid)*grid + posp,grid)
        return all([(posm[i] >= my_box[i,0] and posm[i] < my_box[i,1]) or\
                    (posp[i] >= my_box[i,0] and posp[i] < my_box[i,1]) for i in range(3)])

    for pos, spec in zip(positions, species):
        c_jit = 0
        box_lim = [0,int(grid_np[0]/2)+5,grid_np[0]]
        for ibox in range(2):
            for jbox in range(2):
                for kbox in range(2):
                    my_box = np.zeros([3,2])
                    my_box[:,1] = grid
                    my_box[0,0] = box_lim[ibox]
                    my_box[0,1] = box_lim[ibox+1]
                    my_box[1,0] = box_lim[jbox]
                    my_box[1,1] = box_lim[jbox+1]
                    my_box[2,0] = box_lim[kbox]
                    my_box[2,1] = box_lim[kbox+1]
                    my_box = my_box.astype(int)
                    rho_jit = torch.from_numpy(rho[my_box[0,0]:my_box[0,1],my_box[1,0]:my_box[1,1],my_box[2,0]:my_box[2,1]]).double()
                    my_box = torch.from_numpy(my_box).double()
                    if is_in_box(pos, basis[spec]['r_o'], my_box, grid, a):
                        rad, ang = basis_models[spec](pos, unitcell, grid, a, my_box)
                        rsize = rad.size()
                        if not rsize[-1]*rsize[-2]*rsize[-3]: continue
                        c_jit += projector_models[spec](rho_jit,
                            pos, unitcell, grid, a, rad, ang, my_box).detach().numpy()
        c_np = C[spec].pop(0)
        assert np.allclose(c_jit, c_np)

    # assert False


@pytest.mark.benzene_compiled
def test_benzene_compiled():

    benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
    xc.ml.network.compile_model(benzene_nxc, 'benzene.nxc.jit',
        override=True)
    benzene_nxc = xc.neuralxc.NeuralXCJIT('benzene.nxc.jit')

    benzene_traj = ase.io.read(os.path.join(test_dir, 'benzene_test', 'benzene.xyz'), '0')
    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))

    with torch.jit.optimized_execution(should_optimize=True):
        a = np.linalg.norm(unitcell, axis=1) / grid[:3]
        positions = benzene_traj.get_positions() / Bohr
        species = benzene_traj.get_chemical_symbols()
        start = time()
        benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
        V_comp = benzene_nxc.get_V(rho, calc_forces=True)
        forces_comp = V_comp[1][1][:-3]
        V_comp = V_comp[0], V_comp[1][0]
        end = time()

        time_torch = end - start

    benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
    start = time()
    benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
    V_np = benzene_nxc.get_V(rho, calc_forces=True)
    forces_np = V_np[1][1][:-3]
    V_np = V_np[0], V_np[1][0]
    end = time()
    time_classical = end - start
    print('Torch', time_torch)
    print('Classical', time_classical)
    assert np.allclose(V_np[0], V_comp[0])
    assert np.allclose(V_np[1], V_comp[1])
    assert np.allclose(forces_np, forces_comp)

# @pytest.mark.torch
# def test_benzene_torch():
#
#     benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
#
#     benzene_nxc._pipeline.to_torch()
#     benzene_nxc._pipeline.basis_instructions['projector_type'] = \
#         benzene_nxc._pipeline.basis_instructions.get('projector_type','ortho') + '_torch'
#     benzene_nxc._pipeline.symmetrize_instructions['symmetrizer_type'] = \
#         benzene_nxc._pipeline.symmetrize_instructions.get('symmetrizer_type','casimir') + '_torch'
#     benzene_traj = ase.io.read(os.path.join(test_dir, 'benzene_test', 'benzene.xyz'), '0')
#     density_getter = xc.utils.SiestaDensityGetter(binary=True)
#     rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))
#
#     a = np.linalg.norm(unitcell, axis=1) / grid[:3]
#     positions = benzene_traj.get_positions() / Bohr
#     species = benzene_traj.get_chemical_symbols()
#     start = time()
#     benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
#     V_comp = benzene_nxc.get_V(rho, calc_forces=True)
#     forces_comp = V_comp[1][1][:-3]
#     V_comp = V_comp[0], V_comp[1][0]
#     end = time()
#
#     time_torch = end - start
#
#     benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
#     start = time()
#     benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
#     V_np = benzene_nxc.get_V(rho, calc_forces=True)
#     forces_np = V_np[1][1][:-3]
#     V_np = V_np[0], V_np[1][0]
#     end = time()
#     time_classical = end - start
#     print('Torch', time_torch)
#     print('Classical', time_classical)
#     assert np.allclose(V_np[0], V_comp[0])
#     assert np.allclose(V_np[1], V_comp[1])
#     assert np.allclose(forces_np, forces_comp)
# @pytest.mark.torch_energy
# def test_neuralxc_benzene_torch_energy():
#
#     benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
#     benzene_traj = ase.io.read(os.path.join(test_dir, 'benzene_test', 'benzene.xyz'), '0')
#     density_getter = xc.utils.SiestaDensityGetter(binary=True)
#     rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))
#
#     a = np.linalg.norm(unitcell, axis=1) / grid[:3]
#     positions = benzene_traj.get_positions() / Bohr
#     species = benzene_traj.get_chemical_symbols()
#     benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
#
#     start = time()
#     for i in range(1):
#         V_classical = benzene_nxc.get_V(rho)
#     end = time()
#     normal_time = (-start + end)/1
#
#     benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
#     benzene_nxc._pipeline.to_torch()
#     benzene_nxc._pipeline.basis_instructions['projector_type'] = 'ortho_torch'
#     benzene_nxc._pipeline.symmetrize_instructions['symmetrizer_type'] = 'casimir_torch'
#     rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))
#     benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
#
#     rho = torch.from_numpy(rho).double()
#     rho.requires_grad = True
#     unitcell = torch.from_numpy(unitcell).double()
#     grid = torch.from_numpy(grid).double()
#     positions = torch.from_numpy(positions).double()
#     positions.requires_grad = True
#     species = torch.Tensor([getattr(periodictable,s).number for s in species])
#
#     a = torch.from_numpy(a).double()
#
#     U = torch.einsum('ij,i->ij', unitcell, 1/grid)
#     V_cell = torch.det(U)
#
#     class E_predictor(torch.nn.Module):
#
#         def __init__(self):
#             torch.nn.Module.__init__(self)
#
#         def forward(self, C0, C1):
#              C = {'C' : C0, 'H': C1}
#              x = benzene_nxc.symmetrizer(C)
#              for steps in benzene_nxc._pipeline.steps[:]:
#                  x = steps[1].forward(x)
#              return x
#
#     class ModuleBasis(torch.nn.Module):
#         def __init__(self):
#             torch.nn.Module.__init__(self)
#             self.projector = benzene_nxc.projector
#
#         def forward(self, positions, unitcell, grid, a):
#              radials , angulars = self.projector.forward_basis(positions, unitcell, grid, a)
#              return radials, angulars
#
#
#     class ModuleProject(torch.nn.Module):
#         def __init__(self):
#             torch.nn.Module.__init__(self)
#             self.projector = benzene_nxc.projector
#
#         def forward(self, rho, positions, unitcell, grid, a, radials, angulars):
#              x = self.projector.forward_fast(rho, positions, unitcell, grid, a, radials, angulars)
#              return x
#
#     def calc_molecule(positions, basismodels, projectmodels, rho, unitcell, grid, a):
#             pos = {'C': positions[::2], 'H': positions[1::2]}
#             results = {'C': [], 'H':[]}
#             for spec in ['C','H']:
#                 for p in pos[spec]:
#                     radials, angulars = basismodels[spec](p, unitcell, grid, a)
#                     C = projectmodels[spec](rho,p,unitcell, grid,a, radials, angulars)
#                     results[spec].append(C)
#
#             return (torch.stack(results['C']), torch.stack(results['H']))
#
#     basismod = ModuleBasis()
#     projector = ModuleProject()
#     basismodels_comp = {}
#     projectormodels_comp = {}
#
#     #compilation variables
#     unitcell_c = torch.eye(3).double()*4.0
#     grid_c = torch.Tensor([10,10,10]).double()
#     a_c = torch.norm(unitcell_c, dim=1).double() / grid_c
#     pos_c = torch.Tensor([[0, 0, 0]]).double()
#     rho_c = torch.ones(size=(10,10,10)).double()
#
#     # unitcell_c = unitcell
#     # grid_c = grid
#     # a_c = a
#     # pos_c = positions[0:1]
#     # rho_c = rho
#     with torch.jit.optimized_execution(should_optimize=True):
#         for spec in ['C','H']:
#             basismod.projector.set_species(spec)
#             basismodels_comp[spec] = torch.jit.trace(basismod, (pos_c, unitcell_c, grid_c, a_c), optimize=True, check_trace = True)
#             radials , angulars = basismodels_comp[spec](pos_c, unitcell_c, grid_c, a_c)
#             projectormodels_comp[spec] = torch.jit.trace(projector, (rho_c, pos_c, unitcell_c, grid_c, a_c, radials, angulars), optimize=True, check_trace = True)
#
#         C_compiled = calc_molecule(positions, basismodels_comp, projectormodels_comp, rho, unitcell, grid, a)
#
#
#     epred = E_predictor()
#     E = epred(*C_compiled)
#     E.backward()
#     assert np.allclose(V_classical[1],(rho.grad/V_cell).detach().numpy())
#     with torch.jit.optimized_execution(should_optimize=True):
#         compiled_model = torch.jit.trace(epred, (C_compiled[0][3:4],C_compiled[1][3:4]), check_trace = False)
#         rho.grad.zero_()
#         C_compiled = calc_molecule(positions, basismodels_comp, projectormodels_comp, rho, unitcell, grid, a)
#         E_compiled = compiled_model(*C_compiled)
#         E_compiled.backward()
#
#     assert np.allclose(V_classical[1],(rho.grad/V_cell).detach().numpy())
#
# @pytest.mark.torch
# def test_torch_projector_fast():
#     benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
#     benzene_traj = ase.io.read(os.path.join(test_dir, 'benzene_test', 'benzene.xyz'), '0')
#     density_getter = xc.utils.SiestaDensityGetter(binary=True)
#     rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))
#     positions = benzene_traj.get_positions() / Bohr
#     species = benzene_traj.get_chemical_symbols()
#     a = np.linalg.norm(unitcell, axis=1) / grid[:3]
#
#     benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
#     start = time()
#     C = benzene_nxc.projector.get_basis_rep(rho, positions, species)
#     for spec in C:
#         C[spec] *= 0
#     C['C'][0,0] = 1
#     psi_numpy = benzene_nxc.projector.get_V(C, positions, species)
#     C_numpy = tuple(benzene_nxc.projector.get_basis_rep(rho, positions, species).values())
#     benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
#     benzene_nxc._pipeline.to_torch()
#     benzene_nxc._pipeline.basis_instructions['projector_type'] = 'ortho_torch'
#     benzene_nxc._pipeline.symmetrize_instructions['symmetrizer_type'] = 'casimir_torch'
#     benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
#
#     rho = torch.from_numpy(rho).double()
#     rho.requires_grad = True
#     unitcell = torch.from_numpy(unitcell).double()
#     unitcell.requires_grad = False
#     grid = torch.from_numpy(grid).double()
#     positions = torch.from_numpy(positions).double()
#     species = torch.Tensor([getattr(periodictable,s).number for s in species])
#     a = torch.from_numpy(a).double()
#
#     class ModuleBasis(torch.nn.Module):
#         def __init__(self):
#             torch.nn.Module.__init__(self)
#             self.projector = benzene_nxc.projector
#
#         def forward(self, positions, unitcell, grid, a):
#              radials , angulars = self.projector.forward_basis(positions, unitcell, grid, a)
#              return radials, angulars
#
#
#     class ModuleProject(torch.nn.Module):
#         def __init__(self):
#             torch.nn.Module.__init__(self)
#             self.projector = benzene_nxc.projector
#
#         def forward(self, rho, positions, unitcell, grid, a, radials, angulars):
#              x = self.projector.forward_fast(rho, positions, unitcell, grid, a, radials, angulars)
#              return x
#
#     def calc_molecule(positions, basismodels, projectmodels):
#             pos = {'C': positions[::2], 'H': positions[1::2]}
#             results = {'C': [], 'H':[]}
#             for spec in ['C','H']:
#                 for p in pos[spec]:
#                     radials, angulars = basismodels[spec](p, unitcell, grid, a)
#                     C = projectmodels[spec](rho,p,unitcell, grid,a, radials, angulars)
#                     results[spec].append(C)
#
#             return (torch.stack(results['C']), torch.stack(results['H']))
#
#     basismod = ModuleBasis()
#     projector = ModuleProject()
#     basismodels_comp = {}
#     projectormodels_comp = {}
#     with torch.jit.optimized_execution(should_optimize=True):
#         for spec in ['C','H']:
#             basismod.projector.set_species(spec)
#             basismodels_comp[spec] = torch.jit.trace(basismod, (positions[1], unitcell, grid, a), optimize=True, check_trace = True)
#             radials , angulars = basismodels_comp[spec](positions[3], unitcell, grid, a)
#             projectormodels_comp[spec] = torch.jit.trace(projector, (rho, positions[4], unitcell, grid, a, radials, angulars), optimize=True, check_trace = True)
#
#         C_compiled = calc_molecule(positions, basismodels_comp, projectormodels_comp)
#
#     for ct, cn in zip(C_compiled, C_numpy):
#         assert np.allclose(ct.detach().numpy(),cn)
#
#     C_compiled[0][0][0].backward()
#
#     psi_compiled = (rho.grad/benzene_nxc.projector.V_cell).detach().numpy()
#     assert np.allclose(psi_numpy, psi_compiled)
#
#
# @pytest.mark.torch
# def test_torch_projector():
#     benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
#     benzene_traj = ase.io.read(os.path.join(test_dir, 'benzene_test', 'benzene.xyz'), '0')
#     density_getter = xc.utils.SiestaDensityGetter(binary=True)
#     rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))
#     positions = benzene_traj.get_positions() / Bohr
#     species = benzene_traj.get_chemical_symbols()
#     a = np.linalg.norm(unitcell, axis=1) / grid[:3]
#
#     benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
#     start = time()
#     C = benzene_nxc.projector.get_basis_rep(rho, positions, species)
#     for spec in C:
#         C[spec] *= 0
#     C['C'][0,0] = 1
#     psi_numpy = benzene_nxc.projector.get_V(C, positions, species)
#     C_numpy = tuple(benzene_nxc.projector.get_basis_rep(rho, positions, species).values())
#     benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
#     benzene_nxc._pipeline.to_torch()
#     benzene_nxc._pipeline.basis_instructions['projector_type'] = 'ortho_torch'
#     benzene_nxc._pipeline.symmetrize_instructions['symmetrizer_type'] = 'casimir_torch'
#     benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
#
#     rho = torch.from_numpy(rho).double()
#     unitcell = torch.from_numpy(unitcell).double()
#     unitcell.requires_grad = False
#     grid = torch.from_numpy(grid).double()
#     positions = torch.from_numpy(positions).double()
#     species = torch.Tensor([getattr(periodictable,s).number for s in species])
#     a = torch.from_numpy(a).double()
#
#     class Module(torch.nn.Module):
#         def __init__(self):
#             torch.nn.Module.__init__(self)
#             self.projector = benzene_nxc.projector
#
#         def forward(self, rho, positions, species, unitcell, grid, a):
#              x = self.projector(rho, positions, species, unitcell, grid, a)
#              return tuple(x.values())
#
#
#     some_input = unitcell
#     projector = Module()
#     C_torch = projector(rho, positions, species, unitcell, grid, a)
#     with torch.jit.optimized_execution(should_optimize=True):
#         compiled = torch.jit.trace(projector, (rho, positions, species, unitcell, grid, a), optimize=True, check_trace = False)
#         C_compiled = compiled(rho, positions, species, unitcell, grid, a)
#
#     for ct, cn, cc in zip(C_torch, C_numpy, C_compiled):
#         assert np.allclose(ct,cn)
#         assert np.allclose(ct,cc)
#
#     rho.requires_grad = True
#     C_torch = projector(rho, positions, species, unitcell, grid, a)
#     C_torch[0][0][0].backward()
#     psi_torch = (rho.grad/benzene_nxc.projector.V_cell).detach().numpy()
#
#     with torch.jit.optimized_execution(should_optimize=True):
#         rho.grad.zero_()
#         C_torch_comp = compiled(rho, positions, species, unitcell, grid, a)
#         C_torch_comp[0][0][0].backward()
#         psi_compiled = (rho.grad/benzene_nxc.projector.V_cell).detach().numpy()
#
#     assert np.allclose(psi_numpy, psi_torch)
#     assert np.allclose(psi_torch, psi_compiled)
#
#
# @pytest.mark.skipif(not ase_found, reason='requires ase')
# @pytest.mark.torch
# def test_neuralxc_benzene_torch():
#
#     benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
#     benzene_traj = ase.io.read(os.path.join(test_dir, 'benzene_test', 'benzene.xyz'), '0')
#     density_getter = xc.utils.SiestaDensityGetter(binary=True)
#     rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))
#
#     a = np.linalg.norm(unitcell, axis=1) / grid[:3]
#     positions = benzene_traj.get_positions() / Bohr
#     species = benzene_traj.get_chemical_symbols()
#     benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
#
#     start = time()
#     for i in range(1):
#         V_classical = benzene_nxc.get_V(rho)
#     end = time()
#     normal_time = (-start + end)/1
#
#     benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
#     benzene_nxc._pipeline.to_torch()
#     benzene_nxc._pipeline.basis_instructions['projector_type'] = 'ortho_torch'
#     benzene_nxc._pipeline.symmetrize_instructions['symmetrizer_type'] = 'casimir_torch'
#     rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))
#     benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
#
#     rho = torch.from_numpy(rho).double()
#     rho.requires_grad = True
#     unitcell = torch.from_numpy(unitcell).double()
#     grid = torch.from_numpy(grid).double()
#     positions = torch.from_numpy(positions).double()
#     # positions.requires_grad = True
#     species = torch.Tensor([getattr(periodictable,s).number for s in species])
#
#     a = torch.from_numpy(a).double()
#
#     class E_predictor(torch.nn.Module):
#
#         def __init__(self):
#             torch.nn.Module.__init__(self)
#
#         def forward(self, rho, positions, species, unitcell, grid, a):
#              x = benzene_nxc.symmetrizer(benzene_nxc.projector(rho, positions, species, unitcell, grid, a))
#              for steps in benzene_nxc._pipeline.steps[:]:
#                  x = steps[1].forward(x)
#              return x
#
#     epred = E_predictor()
#     E = epred(rho, positions, species, unitcell, grid, a)
#     E.backward()
#     assert np.allclose(V_classical[1],(rho.grad/benzene_nxc.projector.V_cell).detach().numpy())
#     with torch.jit.optimized_execution(should_optimize=True):
#         compiled_model = torch.jit.trace(epred, (rho, positions, species, unitcell, grid, a), check_trace = False)
#         rho.grad.zero_()
#         E_compiled = compiled_model(rho, positions, species, unitcell, grid, a)
#         E_compiled.backward()
#
#     assert np.allclose(V_classical[1],(rho.grad/benzene_nxc.projector.V_cell).detach().numpy())
