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
@pytest.mark.project
@pytest.mark.parametrize('projector_type',[name for name in \
    xc.projector.projector.BaseProjector.get_registry() if not name in ['default','base','pyscf','default_torch']])
def test_density_projector(projector_type):

    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'h2o.RHO'))

    basis_set = {'O': {'n': 2, 'l': 3, 'r_o': 1}, 'H': {'n': 2, 'l': 2, 'r_o': 1.5}, 'projector_type': projector_type}

    density_projector = xc.projector.DensityProjector(unitcell=unitcell, grid=grid, basis_instructions=basis_set)

    positions = np.array([[0.0, 0.0, 0.0], [-0.75846035, -0.59257417, 0.0], [0.75846035, -0.59257417, 0.0]
                          ]) / xc.constants.Bohr

    basis_rep = density_projector.get_basis_rep(rho, positions=positions, species=['O', 'H', 'H'])

    if projector_type in ['ortho','ortho_torch']:
        if save_test_density_projector:
            with open(os.path.join(test_dir, 'h2o_rep.pckl'), 'wb') as file:
                pickle.dump(basis_rep, file)
        else:
            with open(os.path.join(test_dir, 'h2o_rep.pckl'), 'rb') as file:
                basis_rep_ref = pickle.load(file)

            for spec in basis_rep:
                assert np.allclose(basis_rep[spec], basis_rep_ref[spec])


@pytest.mark.fast
@pytest.mark.parametrize("symmetrizer_type",[name for name in \
    xc.symmetrizer.BaseSymmetrizer.get_registry() if not name in ['default','base']])
def test_symmetrizer(symmetrizer_type):
    with open(os.path.join(test_dir, 'h2o_rep.pckl'), 'rb') as file:
        C = pickle.load(file)

    basis_set = {'O': {'n': 2, 'l': 3, 'r_o': 1}, 'H': {'n': 2, 'l': 2, 'r_o': 1.5}}
    symmetrize_instructions = {'basis': basis_set, 'symmetrizer_type': symmetrizer_type}

    symmetrizer = xc.symmetrizer.symmetrizer_factory(symmetrize_instructions)

    D = symmetrizer.get_symmetrized(C)

    if save_test_symmetrizer:
        with open(os.path.join(test_dir, 'h2o_sym_{}.pckl'.format(symmetrizer_type)), 'wb') as file:
            pickle.dump(D, file)
    elif symmetrizer_type == 'casimir':
        with open(os.path.join(test_dir, 'h2o_sym_{}.pckl'.format(symmetrizer_type)), 'rb') as file:
            D_ref = pickle.load(file)

        for spec in D:
            assert np.allclose(D[spec], D_ref[spec])


@pytest.mark.fast
@pytest.mark.parametrize("symmetrizer_type",[name for name in \
    xc.symmetrizer.BaseSymmetrizer.get_registry() if not name in ['default','base']])
def test_symmetrizer_rot_invariance(symmetrizer_type):
    C_list = []
    for i in range(3):
        with open(os.path.join(test_dir, 'h2o_rot{}.pckl'.format(i)), 'rb') as file:
            C_list.append(pickle.load(file))

    basis_set = {'O': {'n': 2, 'l': 3, 'r_o': 1}, 'H': {'n': 2, 'l': 2, 'r_o': 1.5}}
    symmetrize_instructions = {'basis': basis_set, 'symmetrizer_type': symmetrizer_type}

    symmetrizer = xc.symmetrizer.symmetrizer_factory(symmetrize_instructions)

    D_list = []
    for C in C_list:
        D_list.append(symmetrizer.get_symmetrized(C))

    for D in D_list[1:]:
        for spec in D:
            assert np.allclose(D[spec], D_list[0][spec], rtol=1e-3, atol=1e-4)


@pytest.mark.fast
@pytest.mark.parametrize("symmetrizer_type",[name for name in \
    xc.symmetrizer.BaseSymmetrizer.get_registry() if not name in ['default','base']])
def test_symmetrizer_rot_invariance_synthetic(symmetrizer_type):
    with open(os.path.join(test_dir, 'rotated_synthetic.pckl'), 'rb') as file:
        C_list = pickle.load(file)

    basis_set = {'O': {'n': 2, 'l': 3, 'r_o': 1}, 'H': {'n': 2, 'l': 2, 'r_o': 1.5}}
    symmetrize_instructions = {'basis': basis_set, 'symmetrizer_type': symmetrizer_type}

    symmetrizer = xc.symmetrizer.symmetrizer_factory(symmetrize_instructions)

    D_list = []
    for C in C_list:
        D_list.append(symmetrizer.get_symmetrized(C))

    for D in D_list[1:]:
        for spec in D:
            assert np.allclose(D[spec], D_list[0][spec])


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
                         [[xc.ml.transformer.GroupedPCA(n_components=2),
                           os.path.join(test_dir, 'pca1.pckl')],
                          [xc.ml.transformer.GroupedVarianceThreshold(0.005),
                           os.path.join(test_dir, 'var09.pckl')]])
def test_grouped_transformers(transformer, filepath):
    with open(os.path.join(test_dir, 'transformer_in.pckl'), 'rb') as file:
        C = pickle.load(file)

    transformed = transformer.fit_transform(C)
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

    benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
    benzene_traj = ase.io.read(os.path.join(test_dir, 'benzene_test', 'benzene.xyz'), '0')
    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))

    positions = benzene_traj.get_positions() / Bohr
    species = benzene_traj.get_chemical_symbols()
    benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
    V, forces = benzene_nxc.get_V(rho, calc_forces=True)[1]
    V = V / Hartree
    forces = forces / Hartree * Bohr


@pytest.mark.skipif(not ase_found, reason='requires ase')
@pytest.mark.force
@pytest.mark.parametrize('use_delta', [False, True])
def test_force_correction(use_delta):

    benzene_traj = ase.io.read(os.path.join(test_dir, 'benzene_test', 'benzene.xyz'), '0')
    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.DRHO'))
    positions = benzene_traj.get_positions() / Bohr
    species = benzene_traj.get_chemical_symbols()
    if use_delta:
        drho, _, _ = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.DRHO'))
        benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'dbenzene'))
        benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
        benzene_nxc.projector = xc.projector.DeltaProjector(benzene_nxc.projector)
        benzene_nxc.projector.set_constant_density(drho, positions, species)

    else:
        benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
        benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)

    def get_V_shifted(self, rho, unitcell, grid, positions, positions_shifted, species, calc_forces=False):
        """ only defined to calculate basis set contribution to forces with numerical derivatives.
        For finite difference, the descriptors should be calculated using the original positions,
        whereas V will then be built with displaced atoms
        """
        projector = xc.projector.DensityProjector(unitcell=unitcell,
                                                  grid=grid,
                                                  basis_instructions=self._pipeline.get_basis_instructions())

        if use_delta:
            projector = xc.projector.DeltaProjector(projector)
            projector.set_constant_density(drho, positions, species)

        symmetrize_dict = {'basis': self._pipeline.get_basis_instructions()}
        symmetrize_dict.update(self._pipeline.get_symmetrize_instructions())

        symmetrizer = xc.symmetrizer.symmetrizer_factory(symmetrize_dict)

        C = projector.get_basis_rep(rho, positions=positions, species=species)

        D = symmetrizer.get_symmetrized(C)
        dEdC = symmetrizer.get_gradient(self._pipeline.get_gradient(D))
        E = self._pipeline.predict(D)[0]

        return E, projector.get_V(dEdC, positions=positions_shifted, species=species, calc_forces=calc_forces, rho=rho)

    V, forces = benzene_nxc.get_V(rho, calc_forces=True)[1]
    forces = forces[:-3]  # no stress

    assert np.allclose(np.sum(forces, axis=0), np.zeros(3), atol=1e-6)
    for incr_atom in [0, 1, 2, 3]:
        for incr_dx in range(3):
            incr = 0.00001
            incr_idx = 1
            pp = np.array(positions)
            pm = np.array(pp)
            pp[incr_atom, incr_idx] += incr
            pm[incr_atom, incr_idx] -= incr

            Vp = get_V_shifted(benzene_nxc, rho, unitcell, grid, positions, pp, species)[1]
            Vm = get_V_shifted(benzene_nxc, rho, unitcell, grid, positions, pm, species)[1]

            dv = (unitcell[0, 0] / grid[0])**3
            fp = dv * np.sum(Vp * rho)
            fm = dv * np.sum(Vm * rho)

            forces_fd = (fp - fm) / (2 * incr)
            assert np.allclose(-forces_fd, forces[incr_atom, incr_idx], atol=incr)


@pytest.mark.skipif(not ase_found, reason='requires ase')
@pytest.mark.parallel
@pytest.mark.parametrize('use_delta', [False, True])
def test_parallel(use_delta):

    benzene_traj = ase.io.read(os.path.join(test_dir, 'benzene_test', 'benzene.xyz'), '0')
    # Break symmetry
    positions = benzene_traj.get_positions()
    positions[0, 1] += 0.05
    positions[3, 1] += 0.05
    benzene_traj.set_positions(positions)
    density_getter = xc.utils.SiestaDensityGetter(binary=True)
    rho, unitcell, grid = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.RHOXC'))
    positions = benzene_traj.get_positions() / Bohr
    species = benzene_traj.get_chemical_symbols()
    if use_delta:
        drho, _, _ = density_getter.get_density(os.path.join(test_dir, 'benzene_test', 'benzene.DRHO'))
        rho_const = (rho - drho)
        benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'dbenzene'))
        benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)
        benzene_nxc.projector = xc.projector.DeltaProjector(benzene_nxc.projector)
        benzene_nxc.projector.set_constant_density(rho_const, positions, species)

    else:
        benzene_nxc = xc.NeuralXC(os.path.join(test_dir, 'benzene_test', 'benzene'))
        benzene_nxc.initialize(unitcell=unitcell, grid=grid, positions=positions, species=species)

    V_serial = benzene_nxc.get_V(rho, calc_forces=False)[1]
    benzene_nxc.max_workers = 4
    V_parallel = benzene_nxc.get_V(rho, calc_forces=False)[1]

    assert np.allclose(V_serial, V_parallel, atol=1e-6, rtol=1e-5)
