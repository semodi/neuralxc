"""
Unit and regression test for the neuralxc package.
"""

import copy
import os
import sys
from abc import ABC, abstractmethod
from time import time

import ase
import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import periodictable
import pytest
import torch

# Import package, test suite, and other packages as needed
import neuralxc as xc
from neuralxc.constants import Bohr, Hartree

test_dir = os.path.dirname(os.path.abspath(__file__))

save_test_density_projector = False
save_siesta_density_getter = False
save_test_symmetrizer = False
save_grouped_transformer = False


@pytest.mark.mybox
def test_mybox():

    benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene.jit'))
    benzene_traj = ase.io.read(os.path.join(test_dir, 'benzene_test', 'benzene.xyz'), '0')
    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))
    grid_np = np.array(grid)
    positions = benzene_traj.get_positions() / Bohr
    # Break symmetries
    positions[:, 0] += 0.02
    positions[:, 1] += 0.01
    positions[:, 2] += 0.12
    species = benzene_traj.get_chemical_symbols()
    a = np.linalg.norm(unitcell, axis=1) / grid[:3]
    benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
    basis_models = benzene_nxc.basis_models
    projector_models = benzene_nxc.projector_models

    unitcell = torch.from_numpy(unitcell).double()
    grid = torch.from_numpy(grid).double()
    positions = torch.from_numpy(positions).double()
    a = torch.from_numpy(a).double()

    for pos, spec in zip(positions, species):
        c_jit = 0
        box_lim = [0, int(grid_np[0] / 2) + 5, grid_np[0]]
        for ibox in range(2):
            for jbox in range(2):
                for kbox in range(2):
                    my_box = np.zeros([3, 2])
                    my_box[:, 1] = grid
                    my_box[0, 0] = box_lim[ibox]
                    my_box[0, 1] = box_lim[ibox + 1]
                    my_box[1, 0] = box_lim[jbox]
                    my_box[1, 1] = box_lim[jbox + 1]
                    my_box[2, 0] = box_lim[kbox]
                    my_box[2, 1] = box_lim[kbox + 1]
                    my_box = my_box.astype(int)
                    rho_jit = torch.from_numpy(rho[my_box[0, 0]:my_box[0, 1], my_box[1, 0]:my_box[1, 1],
                                                   my_box[2, 0]:my_box[2, 1]]).double()
                    my_box = torch.from_numpy(my_box).double()
                    rad, ang, box = basis_models[spec](pos, unitcell, grid, my_box)
                    rsize = rad.size()
                    if not rsize[-1]: continue
                    c_jit += projector_models[spec](rho_jit, pos, unitcell, grid, rad, ang, box).detach().numpy()
    assert np.allclose(c_jit, np.load(os.path.join(test_dir, 'my_box_ref.npy')))


@pytest.mark.torch
def test_stress():

    benzene_nxc = xc.neuralxc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene.jit'))

    benzene_traj = ase.io.read(os.path.join(test_dir, 'benzene_test', 'benzene.xyz'), '0')
    density_getter = xc.utils.SiestaDensityGetter(binary=True)

    rho, unitcell_true, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))
    positions = benzene_traj.get_positions() / Bohr
    species = benzene_traj.get_chemical_symbols()

    unitcell = np.array(unitcell_true)
    benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
    V_comp = benzene_nxc.get_V(rho, calc_forces=True)
    stress = V_comp[1][1][-3:]
    forces = V_comp[1][1][:-3]

    stress_diag = []
    dx = 0.0001
    for ij in range(3):
        energies = []
        for ix in [-1, 1]:
            unitcell = np.array(unitcell_true)
            unitcell[ij, ij] *= (1 + dx * ix)
            positions[:, ij] *= (1 + dx * ix)
            benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
            V_comp = benzene_nxc.get_V(rho, calc_forces=True)
            forces_comp = V_comp[1][1][:-3]
            V_comp = V_comp[0], V_comp[1][0]
            energies.append(V_comp[0])
            positions[:, ij] /= (1 + dx * ix)

        V_ucell = np.linalg.det(unitcell_true)
        stress_xx = (energies[1] - energies[0]) / dx * .5 / V_ucell
        stress_diag.append(stress_xx)

    print('Finite diff ', stress_diag)
    print('Exact ', np.diag(stress))
    assert np.allclose(stress_diag, np.diag(stress))
