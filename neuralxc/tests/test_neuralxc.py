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
try:
    import pyscf
    pyscf_found = True
except ModuleNotFoundError:
    pyscf_found = False

test_dir = os.path.dirname(os.path.abspath(__file__))

save_siesta_density_getter = False
save_test_symmetrizer = False
save_grouped_transformer = False


@pytest.mark.fast
def test_siesta_density_getter():

    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'h2o.RHO'))

    results = {'rho_sum': np.sum(rho), 'rho_norm': np.linalg.norm(rho.flatten()), 'unitcell': unitcell, 'grid': grid}

    if save_siesta_density_getter:
        with open(os.path.join(test_dir, 'h2o_dens.pckl'), 'wb') as file:
            pickle.dump(results, file)
    else:
        with open(os.path.join(test_dir, 'h2o_dens.pckl'), 'rb') as file:
            results_ref = pickle.load(file)
        for key in results:
            assert np.allclose(results_ref[key], results[key])


@pytest.mark.fast
def test_formatter():
    with open(os.path.join(test_dir, 'h2o_rep.pckl'), 'rb') as file:
        C = pickle.load(file)
    basis_set = {'O': {'n': 2, 'l': 3, 'r_o': 1}, 'H': {'n': 2, 'l': 2, 'r_o': 1.5}}
    formatter = xc.formatter.Formatter(basis_set)
    C_dict = formatter.inverse_transform(C)
    C_id = formatter.transform(C_dict)
    for spec in C:
        assert np.allclose(C_id[spec], C[spec])
    formatter.fit(C_dict)
    C_id = formatter.transform(C_dict)
    for spec in C:
        assert np.allclose(C_id[spec], C[spec])


@pytest.mark.fast
@pytest.mark.parametrize(['transformer', 'filepath'],
                         [  [xc.ml.transformer.GroupedStandardScaler(),
                           os.path.join(test_dir, 'scaler.pckl')],
                          [xc.ml.transformer.GroupedVarianceThreshold(0.005),
                           os.path.join(test_dir, 'var09.pckl')]])
def test_grouped_transformers(transformer, filepath):

    for use_torch in [False,True] if torch_found else [False]:
        with open(os.path.join(test_dir, 'transformer_in.pckl'), 'rb') as file:
            C = pickle.load(file)

        transformer.fit(C)
        transformed = transformer.transform(C)

        if save_grouped_transformer:
            with open(filepath, 'wb') as file:
                pickle.dump(transformed, file)
        else:
            with open(filepath, 'rb') as file:
                ref = pickle.load(file)
            for spec in transformed:
                assert np.allclose(transformed[spec], ref[spec])


def test_species_grouper():
    with open(os.path.join(test_dir, 'h2o_rep.pckl'), 'rb') as file:
        C = pickle.load(file)

    C = [{spec: C[spec].reshape(1, -1, C[spec].shape[-1]) for spec in C}]
    basis_set = {'O': {'n': 2, 'l': 3, 'r_o': 1}, 'H': {'n': 2, 'l': 2, 'r_o': 1.5}}
    species_grouper = xc.formatter.SpeciesGrouper(basis_set, ['OHH'])
    re_grouped = species_grouper.transform(species_grouper.inverse_transform(C, np.array([[0]])))[0]
    re_grouped = re_grouped[0]
    C = C[0]
    for spec in C:
        assert np.allclose(C[spec], re_grouped[spec])


@pytest.mark.skipif(not ase_found, reason='requires ase')
@pytest.mark.realspace
def test_neuralxc_benzene():

    benzene_nxc = xc.NeuralXCJIT(os.path.join(test_dir, 'benzene_test', 'benzene.jit'))
    benzene_traj = ase.io.read(os.path.join(test_dir, 'benzene_test', 'benzene.xyz'), '0')
    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))

    positions = benzene_traj.get_positions() / Bohr
    species = benzene_traj.get_chemical_symbols()
    benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
    V, forces = benzene_nxc.get_V(rho, calc_forces=True)[1]
    V = V / Hartree
    forces = forces / Hartree * Bohr

    assert np.allclose(V,np.load(os.path.join(test_dir, 'benzene_test', 'V_benzene.npy')))
    assert np.allclose(forces[:-3],np.load(os.path.join(test_dir, 'benzene_test', 'forces_benzene.npy'))[:-3])
