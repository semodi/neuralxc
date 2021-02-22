""" Module that implements a Behler-Parinello type neural network
"""

import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from neuralxc.formatter import atomic_shape
from matplotlib import pyplot as plt
import math
from collections import namedtuple
import h5py
import json
from ase.io import read
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.pipeline import Pipeline
import dill as pickle
import shutil
import copy
import torch
from ..projector import DensityProjector
from ..symmetrizer import Symmetrizer
TorchModule = torch.nn.Module
# import tensorflow


def convert_torch_wrapper(func):
    def wrapped_func(X, *args, **kwargs):
        X = torch.from_numpy(X)
        Y = func(X, *args, **kwargs)
        return Y.detach().numpy()

    return wrapped_func


# class NXCPipeline(Pipeline,torch.nn.Module):
class NXCPipeline(Pipeline):
    def __init__(self, steps, basis_instructions, symmetrize_instructions):
        """ Class that extends the scikit-learn Pipeline by adding get_gradient
        and save methods. The save method is necessary if the final estimator
        is a tensorflow neural network as those cannot be pickled.

        Parameters
        -----------

        steps: list
            List of Transformers with final step being an Estimator
        basis_instructions: dict
            Dictionary containing instructions for the projector.
            Example {'C':{'n' : 4, 'l' : 2, 'r_o' : 3}}
        symmetrize_instructions: dict
            Instructions for symmetrizer.
            Example {symmetrizer_type: 'casimir'}
        """
        # torch.nn.Module.__init__(self)
        self.basis_instructions = basis_instructions
        self.symmetrize_instructions = symmetrize_instructions
        self.steps = steps
        super().__init__(steps)

    def get_symmetrize_instructions(self):
        return self.symmetrize_instructions

    def get_basis_instructions(self):
        return self.basis_instructions

    def start_at(self, step_idx):
        """ Return a new NXCPipeline containing a subset of steps of the
        original NXCPipeline

        Parameters
        ----------
        step_idx: int
            Use all steps following and including step with index step_idx
        """

        return NXCPipeline(self.steps[step_idx:],
                           basis_instructions=self.basis_instructions,
                           symmetrize_instructions=self.symmetrize_instructions)

    def save(self, path, override=False, npmodel=False):
        """ Save entire pipeline to disk.

        Parameters
        ----------
        path: string
            Directory in which to store pipeline
        override: bool
            If directory already exists, only save and override if this
            is set to True

        """
        if os.path.isdir(path):
            if not override:
                raise Exception('Model already exists, set override = True')
            else:
                shutil.rmtree(path)
                os.mkdir(path)
        else:
            os.mkdir(path)

        pickle.dump([self.steps, self.basis_instructions, self.symmetrize_instructions],
                    open(os.path.join(path, 'pipeline.pckl'), 'wb'))

    def to_torch(self):
        for step_idx, _ in enumerate(self.steps):
            self.steps[step_idx][1].to_torch()

    def forward(self, X):
        for steps in self.steps:
            X = steps[1].forward(X)
        return X


def load_pipeline(path):
    """ Load a NXCPipeline from the directory specified in path
    """
    steps, basis_instructions, symmetrize_instructions = \
        pickle.load(open(os.path.join(path, 'pipeline.pckl'), 'rb'))
    return NXCPipeline(steps, basis_instructions, symmetrize_instructions)


def compile_energy(model, C, outpath, override):
    class E_predictor(TorchModule):
        def __init__(self, species, model):
            TorchModule.__init__(self)
            self.species = species
            steps = [model.symmetrizer] + [step[1] for step in model.steps]
            steps[-1] = steps[-1]._network
            self.model = torch.nn.Sequential(*steps)

        def forward(self, *args):
            C = {spec: c for spec, c in zip(self.species, args)}
            return self.model(C)

    e_models = {}
    with torch.jit.optimized_execution(should_optimize=True):
        for spec in C:
            c = torch.from_numpy(C[spec])
            epred = E_predictor(spec, model)
            e_models[spec] = torch.jit.trace(epred, c, optimize=True, check_trace=False)

    try:
        os.mkdir(outpath)
    except FileExistsError:
        if override:
            shutil.rmtree(outpath)
            os.mkdir(outpath)
        else:
            raise Exception('Model exists, set override = True to save at this location')

    for spec in C:
        torch.jit.save(e_models[spec], outpath + '/xc_' + spec)
        open(outpath + '/bas.json', 'w').write(json.dumps(model.basis_instructions))


def compile_projector(projector):
    class ModuleBasis(TorchModule):
        def __init__(self, projector):
            TorchModule.__init__(self)
            self.projector = projector

        def forward(self, positions, unitcell, grid, my_box):
            # positions = torch.einsum('...i,ij->...j',positions, unitcell)
            return self.projector.forward_basis(positions, unitcell, grid, my_box)

    class ModuleProject(TorchModule):
        def __init__(self, projector):
            TorchModule.__init__(self)
            self.projector = projector

        def forward(self, rho, positions, unitcell, grid, radials, angulars, my_box):
            # positions = torch.einsum('...i,ij->...j',positions, unitcell)
            return self.projector.forward_fast(rho, positions, unitcell, grid, radials, angulars, my_box)

    unitcell_c = np.eye(3) * 5.0
    grid_c = np.array([9, 9, 9])
    my_box = np.array([[0, 9]] * 3)
    a_c = np.linalg.norm(unitcell_c, axis=1) / grid_c
    pos_c = np.array([[0, 0, 0]])
    rho_c = np.ones(shape=grid_c)
    species = []
    for spec in projector.basis:
        if len(spec) < 3:
            species.append(spec)

    unitcell_c = torch.from_numpy(unitcell_c).double()
    grid_c = torch.from_numpy(grid_c).double()
    a_c = torch.from_numpy(a_c).double()
    pos_c = torch.from_numpy(pos_c).double()
    rho_c = torch.from_numpy(rho_c).double()
    my_box = torch.from_numpy(my_box).double()
    basismod = ModuleBasis(projector)
    projector = ModuleProject(projector)

    basis_models = {}
    projector_models = {}
    with torch.jit.optimized_execution(should_optimize=True):
        for spec in species:
            basismod.projector.set_species(spec)
            basis_models[spec] = torch.jit.trace(basismod, (pos_c, unitcell_c, grid_c, my_box),
                                                 optimize=True,
                                                 check_trace=True)
            radials, angulars, box = basis_models[spec](pos_c, unitcell_c, grid_c, my_box)
            projector_models[spec] = torch.jit.trace(projector,
                                                     (rho_c, pos_c, unitcell_c, grid_c, radials, angulars, box),
                                                     optimize=True,
                                                     check_trace=False)
            C = projector_models[spec](rho_c, pos_c, unitcell_c, grid_c, radials, angulars, box).unsqueeze(0)

    return basis_models, projector_models


def compile_model(model, outpath, override=False):
    class E_predictor(TorchModule):
        def __init__(self, species, model):
            TorchModule.__init__(self)
            self.species = species
            steps = [model.symmetrizer] + [step[1] for step in model.steps]
            steps[-1] = steps[-1]._network
            self.model = torch.nn.Sequential(*steps)

        def forward(self, *args):
            C = {spec: c for spec, c in zip(self.species, args)}
            return self.model(C)

    class ModuleBasis(TorchModule):
        def __init__(self, projector):
            TorchModule.__init__(self)
            self.projector = projector

        def forward(self, positions, unitcell, grid, my_box):
            # positions = torch.einsum('...i,ij->...j',positions, unitcell)
            return self.projector.forward_basis(positions, unitcell, grid, my_box)

    class ModuleProject(TorchModule):
        def __init__(self, projector):
            TorchModule.__init__(self)
            self.projector = projector

        def forward(self, rho, positions, unitcell, grid, radials, angulars, my_box):
            # positions = torch.einsum('...i,ij->...j',positions, unitcell)
            return self.projector.forward_fast(rho, positions, unitcell, grid, radials, angulars, my_box)

    unitcell_c = np.eye(3) * 5.0
    grid_c = np.array([9, 9, 9])
    my_box = np.array([[0, 9]] * 3)
    a_c = np.linalg.norm(unitcell_c, axis=1) / grid_c
    pos_c = np.array([[0, 0, 0]])
    rho_c = np.ones(shape=grid_c)

    species = []
    basis_instructions = model.basis_instructions
    for spec in basis_instructions:
        if len(spec) < 3:
            species.append(spec)

    model.symmetrize_instructions.update({'basis': model.basis_instructions})
    model.symmetrizer = Symmetrizer(model.symmetrize_instructions)
    try:
        projector = DensityProjector(basis_instructions=basis_instructions, unitcell=unitcell_c, grid=grid_c)
    except TypeError:
        try:
            rho_c = np.array([1, 2, 3])
            projector = DensityProjector(basis_instructions=basis_instructions,
                                         grid_coords=unitcell_c,
                                         grid_weights=grid_c)
            model.symmetrize_instructions.update(projector.symmetrize_instructions)
            model.symmetrizer = Symmetrizer(model.symmetrize_instructions)
        except TypeError:
            C = {}
            for spec in species:
                n = basis_instructions[spec]['n']
                l = basis_instructions[spec]['l']
                C[spec] = np.ones([1, n * l**2])
            compile_energy(model, C, outpath, override)
            return 0

    model.projector = projector

    unitcell_c = torch.from_numpy(unitcell_c).double()
    grid_c = torch.from_numpy(grid_c).double()
    a_c = torch.from_numpy(a_c).double()
    pos_c = torch.from_numpy(pos_c).double()
    rho_c = torch.from_numpy(rho_c).double()
    my_box = torch.from_numpy(my_box).double()
    basismod = ModuleBasis(model.projector)
    projector = ModuleProject(model.projector)

    basis_models = {}
    projector_models = {}
    e_models = {}
    with torch.jit.optimized_execution(should_optimize=True):
        for spec in species:
            basismod.projector.set_species(spec)
            basis_models[spec] = torch.jit.trace(basismod, (pos_c, unitcell_c, grid_c, my_box),
                                                 optimize=True,
                                                 check_trace=True)
            radials, angulars, box = basis_models[spec](pos_c, unitcell_c, grid_c, my_box)
            projector_models[spec] = torch.jit.trace(projector,
                                                     (rho_c, pos_c, unitcell_c, grid_c, radials, angulars, box),
                                                     optimize=True,
                                                     check_trace=True)
            C = projector_models[spec](rho_c, pos_c, unitcell_c, grid_c, radials, angulars, box).unsqueeze(0)
            epred = E_predictor(spec, model)
            e_models[spec] = torch.jit.trace(epred, C, optimize=True, check_trace=False)

    try:
        os.mkdir(outpath)
    except FileExistsError:
        if override:
            shutil.rmtree(outpath)
            os.mkdir(outpath)
        else:
            raise Exception('Model exists, set override = True to save at this location')

    for spec in species:
        torch.jit.save(basis_models[spec], outpath + '/basis_' + spec)
        torch.jit.save(projector_models[spec], outpath + '/projector_' + spec)
        torch.jit.save(e_models[spec], outpath + '/xc_' + spec)


class NetworkEstimator(BaseEstimator):

    allows_threading = False

    def __init__(self,
                 n_nodes,
                 n_layers,
                 b,
                 alpha=0.01,
                 max_steps=20001,
                 test_size=0.0,
                 valid_size=0.2,
                 random_seed=None,
                 batch_size=0,
                 activation='sigmoid',
                 optimizer=None,
                 target_loss=-1):
        """ Estimator wrapper for the tensorflow based Network class which
        implements a Behler-Parinello type neural network
        """
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.b = b
        self.alpha = alpha
        self.max_steps = max_steps
        self.test_size = test_size
        self.valid_size = valid_size
        self.random_seed = random_seed
        self.path = None
        self._network = None
        self.batch_size = batch_size
        self.activation = activation
        self.optimizer = optimizer
        self.target_loss = target_loss
        self.verbose = False
        self.fitted = False

    def get_params(self, *args, **kwargs):
        return {
            'n_nodes': self.n_nodes,
            'n_layers': self.n_layers,
            'b': self.b,
            'alpha': self.alpha,
            'max_steps': self.max_steps,
            'test_size': self.test_size,
            'valid_size': self.valid_size,
            'random_seed': self.random_seed,
            'batch_size': self.batch_size,
            'activation': self.activation,
            'optimizer': self.optimizer,
            'target_loss': self.target_loss,
        }

    def build_network(self):
        print('building network')
        self._network = Energy_Network(n_layers=self.n_layers, n_nodes=self.n_nodes, activation=self.activation)
        if not self.path is None:
            self._network.restore_model(self.path)

    def fit(self, X, y=None, *args, **kwargs):
        if isinstance(X, tuple):
            y = X[1]
            X = X[0]

        if not isinstance(X, list):
            X = [X]

        if not isinstance(y, np.ndarray) and not y:
            y = [np.zeros(len(list(x.values())[0])) for x in X]

        # X = X[0]

        if not self._network:
            self.build_network()
        self._network.train(X[0],
                            y[0],
                            step_size=self.alpha,
                            max_steps=self.max_steps,
                            b_=self.b,
                            train_valid_split=1 - self.valid_size,
                            optimizer=self.optimizer,
                            random_seed=self.random_seed,
                            batch_size=self.batch_size,
                            target_loss=self.target_loss)
        self.fitted = True

    def predict(self, X, *args, **kwargs):

        if self._network is None:
            self.build_network()

        if isinstance(X, tuple):
            X = X[0]
        if not isinstance(X, list):
            X = [X]

        predictions = self._network.predict(X[0])
        return predictions

    def score(self, X, y=None, metric='mae'):

        if isinstance(X, tuple):
            y = X[1]
            X = X[0]

        if metric == 'mae':
            metric_function = (lambda x: np.mean(np.abs(x)))
        elif metric == 'rmse':
            metric_function = (lambda x: np.sqrt(np.mean(x**2)))
        else:
            raise Exception('Metric unknown or not implemented')


#         metric_function = (lambda x: np.mean(np.abs(x)))
        scores = []
        if not isinstance(X, list):
            X = [X]
            y = [y]

        for X_, y_ in zip(X, y):
            scores.append(metric_function(self.predict(X_) - y_))

        return -np.mean(scores)

    # def forward(self, X):
    #     return self._network.forward(X)

    def load_network(self, path):
        self.path = path


def train_net(net, dataloader, max_steps=10000, check_point_every=10, lr=1e-3, weight_decay=1e-7):
    # net.train()

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    MIN_RATE = 1e-7
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'min',
                                                           verbose=True,
                                                           patience=10,
                                                           min_lr=MIN_RATE)

    max_epochs = max_steps
    for epoch in range(max_epochs):
        logs = {}
        epoch_loss = 0
        for data in dataloader:
            rho, energy = data
            result = net(rho)
            loss = loss_fn(result, energy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        logs['log loss'] = np.sqrt(epoch_loss / len(dataloader))
        for i, param_group in enumerate(optimizer.param_groups):
            logs['lr_{}'.format(i)] = float(param_group['lr'])

        if logs['lr_0'] <= MIN_RATE:
            return net

        if epoch % check_point_every == 0:
            scheduler.step(epoch_loss / len(dataloader))
            print(logs['log loss'])

    return net


class Dataset(object):
    def __init__(self, rho, energies):
        self.rho = rho
        self.energies = energies.reshape(-1, 1)

    def __getitem__(self, index):

        rho = {}
        for species in self.rho:
            rho[species] = self.rho[species][index]

        energy = self.energies[index]

        return (rho, energy)

    def __len__(self):
        return len(self.energies)


class Energy_Network(torch.nn.Module):
    def __init__(self, n_nodes, n_layers, activation):
        super(Energy_Network, self).__init__()

    def train(self,
              X,
              y,
              step_size=0.01,
              max_steps=50001,
              b_=0,
              verbose=True,
              optimizer=None,
              train_valid_split=0.8,
              random_seed=None,
              batch_size=0,
              target_loss=-1):

        if not hasattr(self, 'species_nets'):
            species_nets = {}
            for spec in X:
                species_nets[spec] = torch.nn.Sequential(
                    torch.nn.Linear(X[spec].shape[-1], 8),
                    torch.nn.GELU(),
                    torch.nn.Linear(8, 8),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(8, 8),
                    torch.nn.GELU(),
                    torch.nn.Linear(8, 1),
                )
            self.species_nets = torch.nn.ModuleDict(species_nets)

        dataset = Dataset(X, y)
        if batch_size == 0:
            batch_size = len(y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        train_net(self, dataloader, max_steps=max_steps, lr=step_size, weight_decay=b_)

    def predict(self, X):
        data_len = 0
        for spec in X:
            data_len = max([data_len, len(X[spec])])

        y = np.zeros(data_len)
        dataset = Dataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_len, shuffle=False)

        for data in dataloader:
            rho, energy = data
            result = self.forward(rho)

        return result.detach().numpy()

    def forward(self, input):
        output = 0
        for spec in input:
            output += self.species_nets[spec](input[spec])
        output = torch.sum(output, dim=-2)

        return output
