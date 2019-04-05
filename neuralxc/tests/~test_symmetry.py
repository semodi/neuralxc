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

test_dir = os.path.dirname(os.path.abspath(__file__))

save_test_density_projector = False
save_siesta_density_getter = False
save_test_symmetrizer = False
save_grouped_transformer = False

@pytest.mark.symmetry
@pytest.mark.fast
def test_density_projector():

    density_getter = xc.utils.SiestaDensityGetter(binary = True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir,'Benzene.RHOXC_fine'))

    basis_set = {
                'C': {'n' : 5, 'l' : 5, 'r_o': 2},
                'H': {'n' : 5, 'l' : 5, 'r_o': 2}
                }

    density_projector = xc.projector.DensityProjector(unitcell, grid, basis_set)
    positions =  ase.io.read(os.path.join(test_dir, 'benzene.xyz'),'0').get_positions()/Bohr
    species =  ase.io.read(os.path.join(test_dir, 'benzene.xyz'),'0').get_chemical_symbols()

    basis_rep = density_projector.get_basis_rep(rho, positions, species)

    r = np.linspace(0,1,100)
    W = density_projector.get_W(1,5)
    radials = density_projector.radials(r,1,W)
    for rad in radials:
        plt.plot(r,rad)
    plt.show()
    symmetrize_instructions = {'basis': basis_set,
                              'symmetrizer_type': 'casimir'}
    symmetrizer = xc.symmetrizer.symmetrizer_factory(symmetrize_instructions)
    basis_rep = symmetrizer.fit_transform(basis_rep)
    # print(basis_rep)
    # varthr = xc.ml.transformer.GroupedVarianceThreshold(1e-5)
    # basis_rep = varthr.fit_transform(basis_rep)
    mask  = {}
    for spec in basis_rep:
        mask = ((np.mean(basis_rep[spec], axis=0) > 1e-2))
        basis_rep[spec] = basis_rep[spec][:, mask]
    print(np.ptp(basis_rep['C'], axis=0)/np.mean(basis_rep['C'], axis=0))
    print(np.ptp(basis_rep['H'], axis=0)/np.mean(basis_rep['H'], axis=0))
    assert False
