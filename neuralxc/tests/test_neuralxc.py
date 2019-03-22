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
test_dir = os.path.dirname(os.path.abspath(__file__))

save_test_density_projector = False
save_siesta_density_getter = False
save_test_symmetrizer = False
save_grouped_transformer = False

@pytest.mark.fast
def test_doc_inherit():

    class ParentA(ABC):


        def __init__(self):
            """
            This is a documentation
            """
            pass

        @abstractmethod
        def test_function(self):
            """
            This is a test documentation
            """
            pass

    class ParentB(ABC):
        def __init__(self):
            """
            This is a documentation
            """
            pass


    class Child(ParentB, ParentA):

        @doc_inherit
        def __init__(self):
            pass
        @doc_inherit
        def test_function(self):
            pass

    child = Child()
    help(child.test_function)

@pytest.mark.fast
def test_siesta_density_getter():

    density_getter = xc.utils.SiestaDensityGetter(binary = True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir,'h2o.RHO'))

    results = {'rho_sum': np.sum(rho), 'rho_norm': np.linalg.norm(rho.flatten()),
                'unitcell': unitcell, 'grid': grid}


    if save_siesta_density_getter:
        with open(os.path.join(test_dir, 'h2o_dens.pckl'),'wb') as file:
            pickle.dump(results, file)
    else:
        with open(os.path.join(test_dir, 'h2o_dens.pckl'),'rb') as file:
            results_ref = pickle.load(file)
        for key in results:
            assert np.allclose(results_ref[key],results[key])

@pytest.mark.fast
def test_density_projector():

    density_getter = xc.utils.SiestaDensityGetter(binary = True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir,'h2o.RHO'))

    basis_set = {
                'O': {'n' : 2, 'l' : 3, 'r_o': 1},
                'H': {'n' : 2, 'l' : 2, 'r_o': 1.5}
                }

    density_projector = xc.projector.DensityProjector(unitcell, grid, basis_set)
    positions =  np.array(
                  [[0.0,         0.0,        0.0],
                  [-0.75846035, -0.59257417, 0.0],
                  [ 0.75846035, -0.59257417, 0.0]]
                  )*xc.constants.Bohr

    basis_rep = density_projector.get_basis_rep(rho, positions, ['O','H','H'])
    print(basis_rep)
    if save_test_density_projector:
        with open(os.path.join(test_dir, 'h2o_rep.pckl'),'wb') as file:
            pickle.dump(basis_rep, file)
    else:
        with open(os.path.join(test_dir, 'h2o_rep.pckl'),'rb') as file:
            basis_rep_ref = pickle.load(file)

        for spec in basis_rep:
            assert np.allclose(basis_rep[spec],basis_rep_ref[spec])


@pytest.mark.fast
@pytest.mark.parametrize("symmetrizer_type",['casimir'])
def test_symmetrizer(symmetrizer_type):
    with open(os.path.join(test_dir, 'h2o_rep.pckl'),'rb') as file:
        C = pickle.load(file)

    basis_set = {
                'O': {'n' : 2, 'l' : 3, 'r_o': 1},
                'H': {'n' : 2, 'l' : 2, 'r_o': 1.5}
                }
    symmetrize_instructions = {'basis': basis_set,
                              'symmetrizer_type': symmetrizer_type}

    symmetrizer = xc.symmetrizer.symmetrizer_factory(symmetrize_instructions)


    D = symmetrizer.get_symmetrized(C)

    if save_test_symmetrizer:
        with open(os.path.join(test_dir, 'h2o_sym_{}.pckl'.format(symmetrizer_type)),'wb') as file:
            pickle.dump(D, file)
    else:
        with open(os.path.join(test_dir, 'h2o_sym_{}.pckl'.format(symmetrizer_type)),'rb') as file:
            D_ref = pickle.load(file)

        for spec in D:
            assert np.allclose(D[spec], D_ref[spec])


@pytest.mark.fast
@pytest.mark.parametrize("symmetrizer_type",['casimir'])
def test_symmetrizer_rot_invariance(symmetrizer_type):
    C_list = []
    for i in range(3):
        with open(os.path.join(test_dir, 'h2o_rot{}.pckl'.format(i)),'rb') as file:
            C_list.append(pickle.load(file))

    basis_set = {
                'O': {'n' : 2, 'l' : 3, 'r_o': 1},
                'H': {'n' : 2, 'l' : 2, 'r_o': 1.5}
                }
    symmetrize_instructions = {'basis': basis_set,
                              'symmetrizer_type': symmetrizer_type}

    symmetrizer = xc.symmetrizer.symmetrizer_factory(symmetrize_instructions)

    D_list = []
    for C in C_list:
        D_list.append(symmetrizer.get_symmetrized(C))

    for D in D_list[1:]:
        for spec in D:
            assert np.allclose(D[spec], D_list[0][spec], rtol=1e-3, atol=1e-4)

@pytest.mark.fast
@pytest.mark.parametrize("symmetrizer_type",['casimir'])
def test_symmetrizer_rot_invariance_synthetic(symmetrizer_type):
    with open(os.path.join(test_dir, 'rotated_synthetic.pckl'),'rb') as file:
            C_list = pickle.load(file)

    basis_set = {
                'O': {'n' : 2, 'l' : 3, 'r_o': 1},
                'H': {'n' : 2, 'l' : 2, 'r_o': 1.5}
                }
    symmetrize_instructions = {'basis': basis_set,
                              'symmetrizer_type': symmetrizer_type}

    symmetrizer = xc.symmetrizer.symmetrizer_factory(symmetrize_instructions)

    D_list = []
    for C in C_list:
        D_list.append(symmetrizer.get_symmetrized(C))

    for D in D_list[1:]:
        for spec in D:
            assert np.allclose(D[spec], D_list[0][spec])

@pytest.mark.fast
@pytest.mark.parametrize("symmetrizer_type",['casimir'])
def test_symmetrizer_gradient(symmetrizer_type):
    """ Synthetic test to see if symmetrizer gradient respects chain rule of
        differentiation
    """
    with open(os.path.join(test_dir, 'h2o_rep.pckl'),'rb') as file:
        C = pickle.load(file)
    basis_set = {
                'O': {'n' : 2, 'l' : 3, 'r_o': 1},
                'H': {'n' : 2, 'l' : 2, 'r_o': 1.5}
                }
    symmetrize_instructions = {'basis': basis_set,
                              'symmetrizer_type': symmetrizer_type}

    symmetrizer = xc.symmetrizer.symmetrizer_factory(symmetrize_instructions)

    def dummy_E(D):
        if 'O' in D:
            return np.linalg.norm(D['O'])
        else:
            return 0


    D = symmetrizer.get_symmetrized(C)

    mod_incr = 0.00001

    # Compute dEdC
    dEdC = {spec : np.zeros_like(C[spec]) for spec in C}
    for mod_spec in C:
        for mod_idx in range(C[mod_spec].shape[-1]):
            for im in [1,1j]:
                Cp =  copy.deepcopy(C)
                Cm =  copy.deepcopy(C)
                Cp[mod_spec][:,mod_idx] += mod_incr*im
                Cm[mod_spec][:,mod_idx] -= mod_incr*im
                Dp = symmetrizer.get_symmetrized(Cp)
                Dm = symmetrizer.get_symmetrized(Cm)
                dEdC[mod_spec][:,mod_idx] += (dummy_E(Dp) - dummy_E(Dm))/(2*mod_incr)*im

    # Compute dEdD -> dEdC_chain
    dEdD = {spec : np.zeros_like(D[spec]) for spec in D}
    for mod_spec in D:
        for mod_idx in range(D[mod_spec].shape[-1]):
            Dp =  copy.deepcopy(D)
            Dm =  copy.deepcopy(D)
            Dp[mod_spec][:,mod_idx] += mod_incr
            Dm[mod_spec][:,mod_idx] -= mod_incr

            dEdD[mod_spec][:,mod_idx] = (dummy_E(Dp) - dummy_E(Dm))/(2*mod_incr)

    symmetrizer.get_symmetrized(C)
    dEdC_chain = symmetrizer.get_gradient(dEdD)

    print(dEdC['O'].round(10))
    print(dEdC_chain['O'].round(10))
    for spec in dEdC:
        assert np.allclose(dEdC[spec],dEdC_chain[spec])

@pytest.mark.fast
def test_formatter():
    with open(os.path.join(test_dir, 'h2o_rep.pckl'),'rb') as file:
        C = pickle.load(file)
    basis_set = {
                'O': {'n' : 2, 'l' : 3, 'r_o': 1},
                'H': {'n' : 2, 'l' : 2, 'r_o': 1.5}
                }
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
@pytest.mark.parametrize(['transformer','filepath'],
                         [[xc.ml.transformer.GroupedPCA(n_components=1),
                          os.path.join(test_dir, 'pca1.pckl')],
                          [xc.ml.transformer.GroupedVarianceThreshold(0.005),
                          os.path.join(test_dir, 'var09.pckl')]])
def test_grouped_transformers(transformer, filepath):
    with open(os.path.join(test_dir, 'h2o_rot2.pckl'),'rb') as file:
        C = pickle.load(file)


    C = {'H' : C['H'].real}

    transformed = transformer.fit_transform(C)
    if save_grouped_transformer:
        with open(filepath, 'wb') as file:
            pickle.dump(transformed, file)
    else:
        with open(filepath, 'rb') as file:
            ref = pickle.load(file)
        assert np.allclose(transformed['H'], ref['H'])

def test_species_grouper():
    with open(os.path.join(test_dir, 'h2o_rep.pckl'),'rb') as file:
        C = pickle.load(file)

    # print(C)
    C = [{spec: C[spec].reshape(1,-1,C[spec].shape[-1]) for spec in C}]
    basis_set = {
                'O': {'n' : 2, 'l' : 3, 'r_o': 1},
                'H': {'n' : 2, 'l' : 2, 'r_o': 1.5}
                }
    species_grouper = xc.formatter.SpeciesGrouper(basis_set, ['OHH'])
    re_grouped = species_grouper.transform(species_grouper.inverse_transform(C,np.array([[0]])))[0]
    print(re_grouped)
    re_grouped = re_grouped[0]
    C = C[0]
    for spec in C:
        assert np.allclose(C[spec],re_grouped[spec])

@pytest.mark.slow
@pytest.mark.parametrize('random_seed',[41,84])
def test_pipeline_gradient(random_seed):
    data = pickle.load(open(os.path.join(test_dir, 'ml_data.pckl'),'rb'))
    basis_set = {
                'C': {'n' : 6, 'l' : 4, 'r_o': 1},
                'H': {'n' : 6, 'l' : 4, 'r_o': 1.5}
                }
    all_species = ['CCCCCCHHHHHH']
    symmetrizer_instructions = {'basis': basis_set,
                     'symmetrizer_type': 'casimir'}

    spec_group = xc.formatter.SpeciesGrouper(basis_set, all_species)
    symmetrizer = xc.symmetrizer.symmetrizer_factory(symmetrizer_instructions)
    var_selector = xc.ml.transformer.GroupedVarianceThreshold(threshold=1e-5)

    estimator = xc.ml.NetworkEstimator(1, 4, [1e-5,1e-5,1e-5,0],
                            alpha=0.001, max_steps = 4001, test_size = 0.0,
                            valid_size = 0.0, random_seed=random_seed)

    pipeline_list = [('spec_group',  spec_group),
                     ('symmetrizer', symmetrizer),
                     ('var_selector', var_selector)]

    pca = xc.ml.transformer.GroupedPCA(n_components= 0.99, svd_solver='full')
    pipeline_list.append(('pca', pca))

    pipeline_list.append(('estimator', estimator))

    ml_pipeline = xc.ml.NXCPipeline(pipeline_list)
    ml_pipeline.fit(data)

    #Subset of data for which to calculate gradient
    x = data[0:1]
    grad_analytic = ml_pipeline.get_gradient(x)

    incr = 0.01
    grad_fd = np.zeros_like(grad_analytic, dtype = complex)
    for ix in range(1, 20): #IMPORTANT: don't alter ix = 0 (system index)
        for im in [1,1j]:
            xp = np.array(x)
            xp[:,ix] += incr*im
            xm = np.array(x)
            xm[:,ix] -= incr*im
            Ep = ml_pipeline.predict(xp)[0]
            Em = ml_pipeline.predict(xm)[0]
            grad_fd[:,ix] += (Ep-Em)/(2*incr)*im

    print(grad_analytic[0,1:20])
    print(grad_fd[0,1:20])
    assert np.allclose(grad_fd[0,1:20], grad_analytic[0,1:20], rtol=1e-2,
                        atol = 1e-3)
