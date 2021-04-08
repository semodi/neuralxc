"""
preprocessor.py

Preprocess electron density (either represented on grid or through density matrix with GTO) and
into projected descriptors used to fit NeuralXC models. Part of training pipeline but not
relevant for deployed models.

"""
import os
from os.path import join as pjoin

import numpy as np
from dask.distributed import Client
from sklearn.base import BaseEstimator, TransformerMixin

from neuralxc.constants import Bohr
from neuralxc.projector import DensityProjector
from neuralxc.utils.density_getter import density_getter_factory


class Preprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, basis_instructions, src_path, atoms, target_path='', num_workers=1):
        """
        Following basis_instructions, applies a suitable DensityProjector to electron
        densities stored to disk
        """
        self.basis_instructions = basis_instructions
        self.src_path = src_path
        self.atoms = atoms
        self.computed_basis = {}
        self.num_workers = num_workers

    def fit(self, X=None, y=None, **kwargs):
        self.client = kwargs.get('client', None)
        return self

    def transform(self, X=None, y=None):
        basis_rep = self.get_basis_rep()
        self.data = basis_rep
        self.computed_basis = self.basis_instructions
        spec_agn = self.basis_instructions.get('spec_agnostic', False)

        unique_systems = np.array([''.join(self.get_chemical_symbols(a)) for a in self.atoms])
        unique_systems = np.unique(unique_systems, axis=0)
        if spec_agn:
            self.species_string = unique_systems[0][0] * max([len(s) for s in unique_systems])
        else:
            self.species_string = ''.join([s for s in unique_systems])
        # === Padding ===

        #Find padded width of data
        width = {}
        for dat, atoms in zip(self.data, self.atoms):
            width[''.join(self.get_chemical_symbols(atoms))] = len(dat)
        #Sanity check
        assert len(unique_systems) == len(width)
        if spec_agn:
            paddedwidth = max([width[key] for key in width])
        else:
            paddedwidth = sum([width[key] for key in width])

        paddedoffset = {}
        cnt = 0
        for key in width:
            if spec_agn:
                paddedoffset[key] = 0
            else:
                paddedoffset[key] = cnt
                cnt += width[key]

        padded_data = np.zeros([len(self.data), paddedwidth])

        for lidx, (dat, atoms) in enumerate(zip(self.data, self.atoms)):
            syskey = ''.join(self.get_chemical_symbols(atoms))
            padded_data[lidx, paddedoffset[syskey]:paddedoffset[syskey] + len(dat)] = dat

        data = padded_data
        if isinstance(X, list) or isinstance(X, np.ndarray):
            data = data[X]
        return data

    def get_basis_rep(self):

        if self.basis_instructions.get('spec_agnostic', False):
            self.get_chemical_symbols = (lambda x: ['X'] * len(x.get_chemical_symbols()))
        else:
            self.get_chemical_symbols = (lambda x: x.get_chemical_symbols())

        if self.num_workers > 1:
            # cluster = LocalCluster(n_workers=1, threads_per_worker=self.num_workers)
            # print(cluster)
            # client = Client(cluster)
            client = Client()

        class FakeClient():
            def map(self, *args):
                return map(*args)

        if self.num_workers == 1:
            client = FakeClient()

        atoms = self.atoms
        extension = self.basis_instructions.get('extension', 'RHOXC')
        if extension[0] != '.':
            extension = '.' + extension

        jobs = []
        for i, system in enumerate(atoms):
            filename = ''
            for file in os.listdir(pjoin(self.src_path, str(i))):
                if file.endswith(extension):
                    filename = file
                    break
            if filename == '':
                raise Exception('Density file not found in ' +\
                    pjoin(self.src_path,str(i)))

            jobs.append([
                pjoin(self.src_path, str(i), filename),
                system.get_positions() / Bohr,
                self.get_chemical_symbols(system)
            ])
        # results = np.array([j.compute(num_workers = self.num_workers) for j in jobs])
        futures = client.map(transform_one, *[[j[i] for j in jobs] for i in range(3)],
                             len(jobs) * [self.basis_instructions])
        if self.num_workers == 1:
            results = list(futures)
        else:
            results = [f.result() for f in futures]
        return results

    def score(self, *args, **kwargs):
        return 0

    def id(self, *args):
        return 1


def transform_one(path, pos, species, basis_instructions):

    density_getter = density_getter_factory(\
        basis_instructions.get('application', 'siesta'),
        binary = basis_instructions.get('binary', True),
        valence = basis_instructions.get('valence', False),
        grad = basis_instructions.get('grad', 0))

    density_dict = density_getter.get_density(path, return_dict=True)
    density_dict.update({'positions': pos, 'species': species})
    projector = DensityProjector(**density_dict, basis_instructions=basis_instructions)
    rho = density_dict.pop('rho')
    basis_rep = projector.get_basis_rep(rho, **density_dict)
    del density_dict
    results = []

    scnt = {spec: 0 for spec in species}
    for spec in species:
        results.append(basis_rep[spec][scnt[spec]])
        scnt[spec] += 1

    results = np.concatenate(results)
    print(path)
    return results
