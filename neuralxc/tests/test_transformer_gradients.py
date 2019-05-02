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
from neuralxc.constants import Hartree
try:
    import ase
    ase_found = True
except ModuleNotFoundError:
    ase_found = False

test_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.fast
@pytest.mark.symmetrizer_gradient
@pytest.mark.parametrize("symmetrizer_type",[name for name in \
    xc.symmetrizer.BaseSymmetrizer.get_registry() if not name in ['default','base']])
def test_symmetrizer_gradient(symmetrizer_type):
    """ Synthetic test to see if symmetrizer gradient respects chain rule of
        differentiation
    """
    with open(os.path.join(test_dir, 'h2o_rep.pckl'), 'rb') as file:
        C = pickle.load(file)
    basis_set = {'O': {'n': 2, 'l': 3, 'r_o': 1}, 'H': {'n': 2, 'l': 2, 'r_o': 1.5}}
    symmetrize_instructions = {'basis': basis_set, 'symmetrizer_type': symmetrizer_type}

    symmetrizer = xc.symmetrizer.symmetrizer_factory(symmetrize_instructions)

    def dummy_E(D):
        if 'O' in D:
            return np.linalg.norm(D['O'])
        else:
            return 0

    D = symmetrizer.get_symmetrized(C)

    mod_incr = 0.00001

    # Compute dEdC
    dEdC = {spec: np.zeros_like(C[spec]) for spec in C}
    for mod_spec in C:
        for mod_idx in range(C[mod_spec].shape[-1]):
            for im in [1, 1j]:
                Cp = copy.deepcopy(C)
                Cm = copy.deepcopy(C)
                Cp[mod_spec][:, mod_idx] += mod_incr * im
                Cm[mod_spec][:, mod_idx] -= mod_incr * im
                Dp = symmetrizer.get_symmetrized(Cp)
                Dm = symmetrizer.get_symmetrized(Cm)
                dEdC[mod_spec][:, mod_idx] += (dummy_E(Dp) - dummy_E(Dm)) / (2 * mod_incr * im)

    # Compute dEdD -> dEdC_chain
    dEdD = {spec: np.zeros_like(D[spec]) for spec in D}
    for mod_spec in D:
        for mod_idx in range(D[mod_spec].shape[-1]):
            Dp = copy.deepcopy(D)
            Dm = copy.deepcopy(D)
            Dp[mod_spec][:, mod_idx] += mod_incr
            Dm[mod_spec][:, mod_idx] -= mod_incr

            dEdD[mod_spec][:, mod_idx] = (dummy_E(Dp) - dummy_E(Dm)) / (2 * mod_incr)

    symmetrizer.get_symmetrized(C)
    dEdC_chain = symmetrizer.get_gradient(dEdD)

    print(dEdC['O'].round(10))
    print(dEdC_chain['O'].round(10))
    for spec in dEdC:
        assert np.allclose(dEdC[spec], dEdC_chain[spec])

@pytest.mark.pipeline
@pytest.mark.parametrize('random_seed', [41, 42])
@pytest.mark.parametrize("symmetrizer_type",[name for name in \
    xc.symmetrizer.BaseSymmetrizer.get_registry() if not name in ['default','base']])
def test_pipeline_gradient(random_seed, symmetrizer_type):
    # data = pickle.load(open(os.path.join(test_dir, 'ml_data.pckl'), 'rb'))
    data = np.load(os.path.join(test_dir, 'ml_data.npy'))
    basis_set = {'C': {'n': 6, 'l': 4, 'r_o': 2}, 'H': {'n': 6, 'l': 4, 'r_o': 2}}
    all_species = ['CCCCCCHHHHHH']
    symmetrizer_instructions = {'basis': basis_set, 'symmetrizer_type': symmetrizer_type}

    spec_group = xc.formatter.SpeciesGrouper(basis_set, all_species)
    symmetrizer = xc.symmetrizer.symmetrizer_factory(symmetrizer_instructions)
    var_selector = xc.ml.transformer.GroupedVarianceThreshold(threshold=1e-10)

    estimator = xc.ml.NetworkEstimator(4,
                                       1, 1e-5  ,
                                       alpha=0.001,
                                       max_steps=1001,
                                       test_size=0.0,
                                       valid_size=0.1,
                                       random_seed=random_seed,
                                       batch_size=50)

    pipeline_list = [('spec_group', spec_group), ('symmetrizer', symmetrizer), ('var_selector', var_selector)]

    # pca = xc.ml.transformer.GroupedPCA(n_components=0.9999999, svd_solver='full')
    # pipeline_list.append(('pca', pca))
    pipeline_list.append(('scaler', xc.ml.transformer.GroupedStandardScaler()))
    pipeline_list.append(('estimator', estimator))


    ml_pipeline = xc.ml.NXCPipeline(pipeline_list,
                                    basis_instructions=basis_set,
                                    symmetrize_instructions=symmetrizer_instructions)
    ml_pipeline.fit(data)

    #Subset of data for which to calculate gradient
    x = data[0:1]
    grad_analytic = ml_pipeline.get_gradient(x)


    grad_fd = np.zeros_like(grad_analytic, dtype=complex)
    for ix in range(1, 20):
        for im in [1, 1j]:
            xp = np.array(x)
            incr = np.mean(np.abs(xp[:,ix]))/1000
            xp[:, ix] += incr * im
            xm = np.array(x)
            xm[:, ix] -= incr * im
            Ep = ml_pipeline.predict(xp)[0]
            Em = ml_pipeline.predict(xm)[0]
            grad_fd[:, ix] += (Ep - Em) / (2 * incr * im)

    print(grad_analytic[0, 1:20])
    print(grad_fd[0, 1:20])
    # assert np.allclose(grad_fd[0, 1:20], grad_analytic[0, 1:20], rtol=1e-4, atol=1e-5)
    assert np.allclose(grad_fd[0, 1:20], grad_analytic[0, 1:20], rtol=1e-3, atol = 1e-10)
    # assert False
