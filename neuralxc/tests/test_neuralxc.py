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

test_dir = os.path.dirname(os.path.abspath(__file__))

save_test_density_projector = False
save_siesta_density_getter = False

def test_doc_inherit():

    class Parent(ABC):


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

    class Child(Parent):

        @doc_inherit
        def __init__(self):
            pass
        @doc_inherit
        def test_function(self):
            pass

    child = Child()
    help(child.test_function)

def test_siesta_density_getter():

    density_getter = xc.utils.SiestaDensityGetter(binary = True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir,'h2o.RHO'))

    results = {'rho_sum': np.sum(rho), 'rho_norm': np.linalg.norm(rho.flatten()),
                'unitcell': unitcell, 'grid': grid}


    if save_siesta_density_getter:
        with open('h2o_dens.pckl','wb') as file:
            pickle.dump(results, file)
    else:
        with open('h2o_dens.pckl','rb') as file:
            results_ref = pickle.load(file)
        for key in results:
            assert np.allclose(results_ref[key],results[key])

def test_density_projector():

    density_getter = xc.utils.SiestaDensityGetter(binary = True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir,'h2o.RHO'))

    basis_set = {
                'O': {'n' : 2, 'l' : 3, 'r_o': 1},
                'H': {'n':2,'l':2,'r_o':1.5}
                }

    density_projector = xc.projector.DensityProjector(unitcell, grid, basis_set)
    positions =  np.array(
                  [[0.0,         0.0,        0.0],
                  [-0.75846035, -0.59257417, 0.0],
                  [ 0.75846035, -0.59257417, 0.0]]
                  )*xc.constants.Bohr

    basis_rep = density_projector.get_basis_rep(rho, positions, ['O','H','H'])
    if save_test_density_projector:
        with open('h2o_rep.pckl','wb') as file:
            pickle.dump(basis_rep, file)
    else:
        with open('h2o_rep.pckl','rb') as file:
            basis_rep_ref = pickle.load(file)

        for this, ref in zip(basis_rep, basis_rep_ref):
            for key in this:
                assert np.allclose(this[key],ref[key])
