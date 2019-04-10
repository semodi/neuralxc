from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from ..utils.density_getter import density_getter_factory
from ..projector import DensityProjector
from ..formatter import atomic_shape, system_shape
from dask import delayed
from ase.io import read
import os
from os.path import join as pjoin
import numpy as np

class Preprocessor(TransformerMixin, BaseEstimator):

    def __init__(self, basis_instructions, src_path, traj_path, target_path):
        self.basis_instructions = basis_instructions
        self.src_path = src_path
        self.traj_path = traj_path
        self.target_path = target_path
        self.computed_basis = {}

    def fit(self, X=None,y=None):
        return self

    def transform(self, X=None, y=None):
        if not self.computed_basis == self.basis_instructions:
            basis_rep = self.get_basis_rep()
            sys_idx = np.array([0]*len(basis_rep)).reshape(-1,1)
            targets = np.load(self.target_path).reshape(-1,1)
            self.data = np.concatenate([sys_idx, basis_rep, targets], axis = -1)
            self.computed_basis = self.basis_instructions

        data = np.array(self.data)
        if isinstance(X, list) or isinstance(X, np.ndarray):
            data = data[X]

        return {'data': data, 'basis_instructions' :self.basis_instructions}

    def get_basis_rep(self):

        atoms = read(self.traj_path,':')
        extension = self.basis_instructions.get('extension','RHOXC')
        if extension[0] != '.':
            extension = '.' + extension

        jobs = []
        for i, system in enumerate(atoms):
            filename = ''
            for file in os.listdir(pjoin(self.src_path,str(i))):
                if file.endswith(extension):
                    filename = file
                    break
            if filename == '':
                raise Exception('Density file not found in ' +\
                    pjoin(self.src_path,str(i)))

            jobs.append(self.transform_one(pjoin(self.src_path,str(i),filename),
                system.get_positions(),
                system.get_chemical_symbols()))
        results = np.array([j.compute() for j in jobs])

        return results

    @delayed
    def transform_one(self, path, pos, species):

        density_getter = density_getter_factory(\
            self.basis_instructions.get('application', 'siesta'),
            binary = self.basis_instructions.get('binary', True))


        rho, unitcell, grid = density_getter.get_density(path)
        projector = DensityProjector(unitcell, grid, self.basis_instructions)

        basis_rep = projector.get_basis_rep(rho, pos, species)

        results = []

        scnt = {spec : 0 for spec in species}
        for spec in species:
            results.append(basis_rep[spec][scnt[spec]])
            scnt[spec] += 1

        results = np.concatenate(results)
        return results
