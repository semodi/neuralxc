import h5py
import json
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from neuralxc.datastructures.hdf5 import *
from neuralxc.formatter import SpeciesGrouper, atomic_shape
from neuralxc.ml.network import NetworkEstimator as NetworkWrapper
from neuralxc.ml.pipeline import NXCPipeline
from neuralxc.ml.transformer import (GroupedStandardScaler, GroupedVarianceThreshold)
from neuralxc.preprocessor import Preprocessor
from neuralxc.symmetrizer import symmetrizer_factory
from neuralxc.utils import ConfigFile

from ..formatter import atomic_shape, expand

__all__ = [
    'E_from_atoms', 'find_attr_in_tree', 'load_sets', 'get_default_pipeline', 'get_grid_cv', 'get_basis_grid',
    'get_preprocessor', 'SampleSelector'
]


def E_from_atoms(traj):

    energies = {}
    for atoms in traj:
        spec = ''.join(atoms.get_chemical_symbols())
        if spec not in energies:
            energies[spec] = []
        energies[spec].append(atoms.get_potential_energy())

    allspecies = np.unique(list(''.join(list(energies))))

    X = np.zeros([len(energies), len(allspecies)])
    y = np.zeros(len(energies))
    for sysidx, syskey in enumerate(energies):
        for sidx, spec in enumerate(allspecies):
            X[sysidx, sidx] = syskey.count(spec)
        y[sysidx] = np.mean(energies[syskey])
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, y)
    offsets = lr.predict(X)
    offsets = dict(zip(energies, offsets))

    energies = []
    for atoms in traj:
        spec = ''.join(atoms.get_chemical_symbols())

        energies.append(atoms.get_potential_energy() - offsets[spec])

    return np.array(energies)


def find_attr_in_tree(file, tree, attr):
    """
    Given the leaf group inside a hdf5 file walk back the tree to
    find a specific attribute

    Parameters
    -----------

    file: h5py.File
        Opened h5py file containing data

    tree: str
        path inside file from which to start looking backwards for attribute

    attr: str
        attribute to look for
"""
    if attr in file[tree].attrs:
        return file[tree].attrs[attr]

    tree_list = tree.split('/')

    for i in range(1, len(tree_list)):
        subtree = '/'.join(tree_list[:-i])
        if attr in file[subtree].attrs:
            return file[subtree].attrs[attr]


def load_sets(datafile, baseline, reference, basis_key='', percentile_cutoff=0):
    """
    Load multiple datasets from hdf5 file

    Parameters
    ----------

    datafile: h5py.File
        File containing data

    baseline: str or list of str
        Group containing baseline datasets including energies and densities

    reference: str or list of str
        Group containing reference dataset (only energy)

    basis_key: str
        Hash to identify basis

    percentile_cutoff: float
        Cutoff this percentage of extreme (in the sense of target value)
        datapoints. Use to remove outliers
    """
    if not isinstance(baseline, list):
        baseline = [baseline]

    if not isinstance(reference, list):
        reference = [reference]

    if not isinstance(percentile_cutoff, list):
        percentile_cutoff = [percentile_cutoff] * len(baseline)

    Xs = []
    ys = []
    max_width = 0

    for sysidx, (bl, ref, perc) in enumerate(zip(baseline, reference, percentile_cutoff)):
        X, y =\
            load_data(datafile, bl, ref, basis_key, perc)
        X = np.concatenate([np.array([sysidx] * len(X)).reshape(-1, 1), X], axis=1)
        max_width = max(max_width, X.shape[1])
        Xs.append(X)
        ys.append(y)

    #pad datasets
    if len(Xs) > 1:
        for i, X in enumerate(Xs):
            if X.shape[1] < max_width:
                Xs[i] = np.concatenate([X, np.zeros([len(X), max_width - X.shape[1]])], axis=1)

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return np.concatenate([X, y.reshape(-1, 1)], axis=1)


def load_data(datafile, baseline, reference, basis_key, percentile_cutoff=0.0, E0=None):
    """
    Load data from hdf5 file

    Parameters
    ----------

    datafile: h5py.File
        File containing data

    baseline: str
        Group containing baseline datasets including energies and densities

    reference: str
        Group containing reference dataset (only energy)

    basis_key: str
        Hash to identify basis

    percentile_cutoff: float
        Cutoff this percentage of extreme (in the sense of target value)
        datapoints. Use to remove outliers

    E0 : float (default: None):
        provide a energy value to subtract from the targets.
        If None tries to find this value as an attribute inside datafile
    """
    no_energy = False
    if f'{baseline}/energy' in datafile:
        data_base = datafile[f'{baseline}/energy']
        data_ref = datafile[f'{reference}/energy']
    else:
        data_base = np.array([0])
        data_ref = np.array([0])
        no_energy = True

    if E0 is None:
        E0_base = find_attr_in_tree(datafile, baseline, 'E0')
        E0_ref = find_attr_in_tree(datafile, reference, 'E0')
    else:
        E0_base = E0
        E0_ref = 0

    if E0_base is None:
        print('Warning: E0 for baseline data not found, setting to 0')
        E0_base = 0
    if E0_ref is None:
        print('Warning: E0 for reference data not found, setting to 0')
        E0_ref = 0

    tar = (data_ref[:] - E0_ref) - (data_base[:] - E0_base)
    tar = tar.real

    if basis_key == '':
        data_base = np.zeros([len(tar), 0])
    else:
        data_base = datafile[f'{baseline}/density/{basis_key}'][:, :]
    if no_energy:
        tar = np.zeros(len(data_base))

    if percentile_cutoff > 0:
        lim1 = np.percentile(tar, percentile_cutoff * 100)
        lim2 = np.percentile(tar, (1 - percentile_cutoff) * 100)
        min_lim, max_lim = min(lim1, lim2), max(lim1, lim2)
        filter = (tar > min_lim) & (tar < max_lim)
    else:
        filter = [True] * len(tar)

    data_base = data_base[filter]
    tar = tar[filter]
    return data_base, tar


def match_hyperparameter(hp, parameters):
    """ Given a partial hyperparameter name hp find the
    corresponding full name in parameters
    """

    matches = [par for par in parameters if hp == par]
    if len(matches) != 1:
        raise ValueError(
            f'{len(matches)} matches found for hyperparameter {hp}. Must be exactly 1'
        )

    return matches[0]


def to_full_hyperparameters(hp, parameters):
    """ Convert partial hyperparameter names to full hp names used
    in GridSearchCV
    """
    full = {}
    for name in hp:
        new_key = f'ml__{match_hyperparameter(name, parameters)}'
        full[new_key] = hp[name]
        if not isinstance(full[new_key], list):
            full[new_key] = [full[new_key]]
    return full


def get_default_pipeline(basis, species, symmetrizer_type='trace', pca_threshold=1, spec_agnostic=False):
    """
    Get the default pipeline containing symmetrizer, variance selector, pca, and
    the final NetworkEstimator
    """
    symmetrizer_instructions = {'basis': basis, 'symmetrizer_type': symmetrizer_type}

    spec_group = SpeciesGrouper(basis, species, spec_agnostic)
    symmetrizer = symmetrizer_factory(symmetrizer_instructions)
    var_selector = GroupedVarianceThreshold(threshold=1e-10)

    estimator = NetworkWrapper(4, 1, 0, alpha=0.001, max_steps=4001, test_size=0.0, valid_size=0, random_seed=None)

    pipeline_list = [
        ('spec_group', spec_group),
        ('symmetrizer', symmetrizer),
        ('var_selector', var_selector),
        ('scaler', GroupedStandardScaler()),
    ]


    pipeline_list.append(('estimator', estimator))

    basis_instructions = basis
    symmetrizer_instructions = {'symmetrizer_type': symmetrizer_type}

    return NXCPipeline(pipeline_list,
                       basis_instructions=basis_instructions,
                       symmetrize_instructions=symmetrizer_instructions)


def get_grid_cv(hdf5, preprocessor, inputfile, spec_agnostic=False):
    if isinstance(preprocessor, str):
        pre = ConfigFile(preprocessor)
    else:
        pre = preprocessor
    inp = json.loads(open(inputfile, 'r').read())

    with h5py.File(hdf5[0], 'r') as datafile:
        if not isinstance(hdf5[1], list):
            hdf5[1] = [hdf5[1]]

        all_species = [
            ''.join(find_attr_in_tree(datafile, set, 'species'))
            for set in hdf5[1]
        ]

    if pre:
        basis = pre['preprocessor']
    else:
        basis = {spec: {'n': 1, 'l': 1, 'r_o': 1} for spec in ''.join(all_species)}
        basis['extension'] = 'RHOXC'

    pipeline = get_default_pipeline(basis,
                                    all_species,
                                    symmetrizer_type=pre.get('symmetrizer_type', 'trace'),
                                    spec_agnostic=spec_agnostic)

    if 'hyperparameters' in inp:
        hyper = inp['hyperparameters']
    else:
        print('No hyperparameters specified, fitting default pipeline to data')

    hyper = to_full_hyperparameters(hyper, pipeline.get_params())

    cv = inp.get('cv', 2)
    n_jobs = inp.get('n_jobs', 1)
    verbose = inp.get('verbose', 1)

    pipe = Pipeline([('ml', pipeline)])
    return GridSearchCV(
        pipe,
        hyper,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=verbose,
        return_train_score=True,
    )


def get_basis_grid(preprocessor):
    """ Give a file containing several basis sets return a grid
    of basis sets that can be used for hyperparameter optimization
    """

    basis = preprocessor['preprocessor']

    from collections import abc

    def nested_dict_iter(nested):
        for key, value in nested.items():
            if isinstance(value, abc.Mapping):
                yield from nested_dict_iter(value)
            else:
                yield key, value

    def nested_dict_build(nested, i):
        select_dict = {}
        for key, value in nested.items():
            if isinstance(value, abc.Mapping):
                select_dict[key] = nested_dict_build(value, i)
            else:
                select_dict[key] = value[i] if isinstance(value, list) else value
        return select_dict

    max_len = 0

    #Check for consistency and build dict mask
    for key, value in nested_dict_iter(basis):
        if isinstance(value, list):
            new_len = len(value)
            if new_len != max_len and max_len != 0:
                raise ValueError('Inconsistent list lengths in basis sets')
            else:
                max_len = new_len

    max_len = max(max_len, 1)
    basis_grid = [nested_dict_build(basis, i) for i in range(max_len)]
    basis_grid = {'preprocessor__basis_instructions': basis_grid}

    return basis_grid


def get_preprocessor(pre, atoms, src_path):
    species = ''.join(atoms[0].get_chemical_symbols())
    for a in atoms:
        species2 = ''.join(a.get_chemical_symbols())
        if species2 != species:
            print('Warning (in get_preprocessor): Dataset not homogeneous')

    basis = {spec: {'n': 1, 'l': 1, 'r_o': 1} for spec in species}
    basis |= pre['preprocessor']
    return Preprocessor(
        basis, src_path, atoms, num_workers=pre.get('n_workers', 1)
    )


class SampleSelector(BaseEstimator):
    def __init__(self, n_instances, random_state=None):
        self._n_instances = n_instances
        self._random_state = random_state

    def fit(self, X, y=None):
        pass

    def predict(self, X):

        if isinstance(X, tuple):
            X = X[0]
        if not isinstance(X, list):
            X = [X]
        picks = [[] * len(X)]
        for idx, key, data in expand(X):
            data = data[0]
            indices = np.zeros(data.shape[:-1], dtype=int)
            indices[:] = np.arange(len(data)).reshape(-1, 1)
            picks[idx] += self.sample_clusters(atomic_shape(data),
                                               indices.flatten(),
                                               self._n_instances,
                                               picked=picks[idx],
                                               random_state=self._random_state)
        picks = picks[0]
        np.random.shuffle(picks)
        return picks[:self._n_instances]

    @staticmethod
    def sample_clusters(data,
                        indices,
                        n_samples,
                        picked=[],
                        cluster_method=KMeans,
                        pca_threshold=0.999,
                        random_state=None):

        clustering = cluster_method(n_samples, random_state=random_state).fit(data)
        labels = clustering.labels_
        centers = clustering.cluster_centers_

        sampled = []
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(data)
        np.random.seed(random_state)
        for center, label in zip(centers, np.unique(labels)):  # Loop over clusters
            _, idx = nbrs.kneighbors(center.reshape(1, -1))
            choice = idx[0]
            if choice not in picked:
                sampled.append(indices[choice])

        return sampled
